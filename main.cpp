#include "ggml.h"

#include "utils.h"
#include "llama.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#endif

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

static const int EOS_TOKEN_ID = 2;

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//
    

static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            _exit(130);
        }
    }
}
#endif

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";

    return s.c_str();
}


void print_info(gpt_params& params, gpt_vocab& vocab, std::vector<gpt_vocab::id>& antiprompt_inp, std::vector<gpt_vocab::id>& embd_inp) {
    fprintf(stderr, "\n");
    fprintf(stderr, "params.n_predict = %d\n", params.n_predict);
    fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    for (int i = 0; i < (int) embd_inp.size(); i++) {
        fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    }
    fprintf(stderr, "\n");
    if (params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#endif

        fprintf(stderr, "%s: interactive mode on.\n", __func__);

        if(antiprompt_inp.size()) {
            fprintf(stderr, "%s: reverse prompt: '%s'\n", __func__, params.antiprompt.c_str());
            fprintf(stderr, "%s: number of tokens in reverse prompt = %zu\n", __func__, antiprompt_inp.size());
            for (int i = 0; i < (int) antiprompt_inp.size(); i++) {
                fprintf(stderr, "%6d -> '%s'\n", antiprompt_inp[i], vocab.id_to_token.at(antiprompt_inp[i]).c_str());
            }
            fprintf(stderr, "\n");
        }

    }
    fprintf(stderr, "sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    if (params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
               " - Press Ctrl+C to interject at any time.\n"
#endif
               " - Press Return to return control to LLaMa.\n"
               " - If you want to submit another line, end your input in '\\'.\n");
    }
    fprintf(stderr, "\n\n");
}

bool detect_reverse_prompt(std::vector<gpt_vocab::id>& antiprompt_inp, std::vector<gpt_vocab::id>& last_n_tokens) {
    return antiprompt_inp.size() && std::equal(antiprompt_inp.rbegin(), antiprompt_inp.rend(), last_n_tokens.rbegin());
}

int read_user_input(gpt_params& params, gpt_vocab& vocab, std::vector<gpt_vocab::id>& embd_inp) {
    bool another_line=true;
    int total_tokens_read = 0;
    while (another_line) {
        fflush(stdout);
        char buf[256] = {0};
        int n_read;
        if(params.use_color) printf(ANSI_BOLD ANSI_COLOR_GREEN);
        if (scanf("%255[^\n]%n%*c", buf, &n_read) <= 0) {
            // presumable empty line, consume the newline
            scanf("%*c");
            n_read=0;
        }
        if(params.use_color) printf(ANSI_COLOR_RESET);

        if (n_read > 0 && buf[n_read-1]=='\\') {
            another_line = true;
            buf[n_read-1] = '\n';
            buf[n_read] = 0;
        } else {
            another_line = false;
            buf[n_read] = '\n';
            buf[n_read+1] = 0;
        }

        std::vector<gpt_vocab::id> line_inp = ::llama_tokenize(vocab, buf, false);
        embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
        
        //fprintf(stderr, "\n\nline_inp.size() = %d\n\n", line_inp.size());
        //fprintf(stderr, "\n\nremaining_tokens = %d\n\n", remaining_tokens);
        total_tokens_read += line_inp.size();
    }
    return total_tokens_read;
}

int run_model(gpt_vocab& vocab, llama_model& model, gpt_params& params, std::mt19937 rng) {
    int n_past = 0;

    const float top_k = params.top_k;
    const float top_p = params.top_p;
    const float temp  = params.temp;
    const float repeat_penalty = params.repeat_penalty;

    const int n_vocab = model.hparams.n_vocab;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;
    std::vector<float> logits;
    std::vector<gpt_vocab::id> embd_inp = ::llama_tokenize(vocab, params.prompt, true);
    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());
    // tokenize the reverse prompt
    std::vector<gpt_vocab::id> antiprompt_inp = ::llama_tokenize(vocab, params.antiprompt, false);
    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    llama_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    int last_n_size = params.repeat_last_n;
    std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    int remaining_tokens = params.n_predict;
    int input_consumed = 0;
    bool input_noecho = false;

    // prompt user immediately after the starting prompt has been loaded
    if (params.interactive_start) {
        is_interacting = true;
    }

    // set the color for the prompt which will be output initially
    if (params.use_color) {
        printf(ANSI_COLOR_YELLOW);
    }

    print_info(params, vocab, antiprompt_inp, embd_inp);
    while (remaining_tokens > 0) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!llama_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                fprintf(stderr, "Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (embd_inp.size() <= input_consumed) {
            // out of user input, sample next token
            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                if (params.ignore_eos) {
                    // set the logit of the eos token to zero to avoid sampling it
                    logits[logits.size() - n_vocab + EOS_TOKEN_ID] = 0;
                }

                id = llama_sample_top_p_top_k(vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_k, top_p, temp, rng);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_noecho = false;

            // decrement remaining sampling budget
            --remaining_tokens;

        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while (embd_inp.size() > input_consumed) {
                embd.push_back(embd_inp[input_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[input_consumed]);
                ++input_consumed;
                if (embd.size() > params.n_batch) {
                    break;
                }
            }

            // reset color to default if we there is no pending user input
            if (!input_noecho && params.use_color && embd_inp.size() == input_consumed) {
                printf(ANSI_COLOR_RESET);
            }
        }

        // display text
        if (!input_noecho) {
            for (auto id : embd) {
                printf("%s", vocab.id_to_token[id].c_str());
            }
            fflush(stdout);
        }

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive && embd_inp.size() <= input_consumed) {
            // check for reverse prompt
            if (detect_reverse_prompt(antiprompt_inp, last_n_tokens)) {
                // currently being interactive
                int n_tokens_read = read_user_input(params, vocab, embd_inp);
                remaining_tokens -= n_tokens_read;
                is_interacting = false;
                input_noecho = true; // do not echo this again
            }
        }

        // end of text token
        if (embd.back() == EOS_TOKEN_ID) {
            fprintf(stderr, " [end of text]\n");
            break;
        }

        if (remaining_tokens <= 0) {
            fprintf(stderr, "No remaining tokens. remaining_tokens = %d", remaining_tokens);
        }
    }


    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        fprintf(stderr, "%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        fprintf(stderr, "%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
    }

    ggml_free(model.ctx);
}

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    llama_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        fprintf(stderr, "test!");
        if (!llama_model_load_fast(params.model, model, vocab, params.n_predict)) {  // TODO: set context from user input ??
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
        fprintf(stderr, "%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
    }

    // Run the model
    run_model(vocab, model, params, rng);

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    return 0;
}
