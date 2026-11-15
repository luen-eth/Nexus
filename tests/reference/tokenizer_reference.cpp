#include "llama.h"
#include <cstdio>
#include <vector>
#include <cstring>
int main(int argc,char**argv){
 if(argc!=3)return 1;llama_backend_init();auto mp=llama_model_default_params();mp.vocab_only=true;mp.n_gpu_layers=0;
 auto* m=llama_model_load_from_file(argv[1],mp);if(!m)return 2;auto* v=llama_model_get_vocab(m);
 std::vector<llama_token> ids(std::strlen(argv[2])*4+1024);
 int n=llama_tokenize(v,argv[2],std::strlen(argv[2]),ids.data(),ids.size(),true,true);if(n<0)return 3;ids.resize(n);
 std::printf("{\"ids\":[");for(int i=0;i<n;i++)std::printf("%s%d",i?",":"",ids[i]);
 std::vector<char> bytes(n*64+1024);int length=llama_detokenize(v,ids.data(),n,bytes.data(),bytes.size(),true,false);if(length<0)return 4;
 std::printf("],\"hex\":\"");for(int i=0;i<length;i++)std::printf("%02x",(unsigned char)bytes[i]);std::printf("\"}\n");
 llama_model_free(m);llama_backend_free();
}
