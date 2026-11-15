#include "llama.h"
#include <cstdio>
int main(int argc,char**argv) {
    if(argc!=2)return 1;
    llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    auto* model=llama_model_load_from_file(argv[1],mp);if(!model)return 2;
    auto cp=llama_context_default_params();cp.n_ctx=32;cp.n_batch=32;cp.n_ubatch=32;cp.n_threads=1;cp.n_threads_batch=1;
    cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* ctx=llama_init_from_model(model,cp);if(!ctx)return 3;
    llama_token tokens[]={3,4,5};auto batch=llama_batch_get_one(tokens,3);
    if(llama_decode(ctx,batch))return 4;
    const auto* logits=llama_get_logits_ith(ctx,-1);auto* vocab=llama_model_get_vocab(model);
    std::printf("[");for(int i=0;i<llama_vocab_n_tokens(vocab);i++) std::printf("%s%.9g",i?",":"",logits[i]);std::printf("]\n");
    llama_free(ctx);llama_model_free(model);llama_backend_free();
}
