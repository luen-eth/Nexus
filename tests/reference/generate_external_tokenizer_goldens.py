from tokenizers import Tokenizer
import json,pathlib,subprocess,jinja2,datetime,os
base=pathlib.Path(os.environ.get('NEXUS_REFERENCE_MODELS',str(pathlib.Path.home()/'.cache/nexus-reference-models')))
corpus=['',' hello','  hello world!','Türkçe İSTANBUL ıi','café cafe\u0301','中文 你好','🙂🚀','a\n\nb\tc','<|im_start|>hello<|im_end|>','<s>hello</s>','hello '*100, "x == 's'", "sep = ' '", "Hello ? ! . , I'm you're"]
messages=[{'role':'system','content':'Be concise.'},{'role':'user','content':'Türkçe 🙂'},{'role':'assistant','content':'Tamam.'},{'role':'user','content':'你好'}]
for name in ['smollm','qwen','llama3']:
 root=base/name;t=Tokenizer.from_file(str(root/'tokenizer.json'));t.no_truncation();t.no_padding();cfg=json.loads((root/'tokenizer_config.json').read_text())
 ctx={**cfg,'messages':messages,'add_generation_prompt':True}
 for key in ['bos_token','eos_token','unk_token','pad_token']:
  if isinstance(ctx.get(key),dict):ctx[key]=ctx[key]['content']
 ctx['raise_exception']=lambda msg:(_ for _ in ()).throw(ValueError(msg))
 ctx['strftime_now']=lambda fmt:datetime.datetime.now(datetime.timezone.utc).strftime(fmt)
 env=jinja2.Environment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols'])
 env.filters['tojson']=lambda value,ensure_ascii=False,indent=None,separators=None,sort_keys=False:json.dumps(value,ensure_ascii=ensure_ascii,indent=indent,separators=separators,sort_keys=sort_keys)
 rendered=env.from_string(cfg['chat_template']).render(ctx)
 cases=[{'text':text,'ids':t.encode(text,add_special_tokens=True).ids,'fragment_ids':t.encode(text,add_special_tokens=False).ids,'decoded':t.decode(t.encode(text,add_special_tokens=True).ids,skip_special_tokens=True)} for text in corpus]
 (root/'golden.json').write_text(json.dumps({'cases':cases,'messages':messages,'rendered':rendered,'chat_ids':t.encode(rendered,add_special_tokens=False).ids},ensure_ascii=False,indent=2))
 print(name,'HF golden ready')
for family in ['llama-spm','llama-bpe','qwen2','bert-bge']:
 path=pathlib.Path(os.environ.get('NEXUS_LLAMA_VOCABS','/tmp/nexus-llama-reference/models'))/f'ggml-vocab-{family}.gguf'
 result=[]
 for text in corpus:
  p=subprocess.run([os.environ.get('NEXUS_LLAMA_TOKENIZER_BIN','/tmp/nexus-tokenizer-reference'),str(path),text],capture_output=True,check=True,text=True)
  d=json.loads(p.stdout);d['text']=text;d['decoded']=bytes.fromhex(d.pop('hex')).decode('utf-8',errors='replace');result.append(d)
 (base/f'gguf-{family}-golden.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(family,'GGUF golden ready')
