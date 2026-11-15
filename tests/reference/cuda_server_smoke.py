import json,pathlib,subprocess,socket,time,urllib.request,urllib.error,concurrent.futures,argparse,tempfile
parser=argparse.ArgumentParser();parser.add_argument('model');args=parser.parse_args()
ROOT=pathlib.Path(__file__).resolve().parents[2];MODEL=str(pathlib.Path(args.model).resolve())
checks=[]
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close();base=f'http://127.0.0.1:{port}'
log=tempfile.TemporaryFile(mode='w+')
proc=subprocess.Popen([str(ROOT/'target/release/nexus-server'),'--model',MODEL,'--host','127.0.0.1','--port',str(port),'--backend','cuda','--kv-quant','none','--ctx-size','256','--batch-size','2','--api-key','reference-only'],stdout=log,stderr=log)
def req(path,body=None,raw=False):
 data=None if body is None else json.dumps(body).encode();headers={'Authorization':'Bearer reference-only'}
 if data is not None:headers['Content-Type']='application/json'
 request=urllib.request.Request(base+path,data=data,headers=headers)
 try:
  with urllib.request.urlopen(request,timeout=120) as r:status,content=r.status,r.read().decode()
 except urllib.error.HTTPError as e:status,content=e.code,e.read().decode()
 return status,content if raw else json.loads(content)
try:
 for _ in range(300):
  if proc.poll() is not None:raise RuntimeError('Server exited; see log')
  try:
   if req('/health')[1].get('model_loaded'):break
  except OSError:pass
  time.sleep(.1)
 else:raise TimeoutError('Server did not load')
 info=req('/v1/model')[1];name=info['id']
 check('Actual CUDA backend',info['backend']=='cuda')
 check('Exact embedded HF tokenizer',info['tokenizer']=='huggingface-json')
 check('Device weights resident',info['device_weight_bytes']>500_000_000)
 chat={'model':name,'messages':[{'role':'user','content':'What is the capital of France?'}],'max_tokens':24,'temperature':0}
 status,response=req('/v1/chat/completions',chat);text=response['choices'][0]['message']['content']
 check('Real model chat',status==200 and 'Paris' in text)
 status,stream=req('/v1/chat/completions',{**chat,'stream':True},raw=True)
 chunks=[json.loads(line[6:]) for line in stream.splitlines() if line.startswith('data: ') and line[6:]!='[DONE]']
 actual=''.join(chunk.get('choices',[{}])[0].get('delta',{}).get('content','') for chunk in chunks)
 check('Chat SSE equals final text',actual==text)
 payload={'model':name,'prompt':'The capital of France is','max_tokens':16,'temperature':0}
 status,response=req('/v1/completions',payload);complete=response['choices'][0]['text']
 stop=complete[:4]
 status,response=req('/v1/completions',{**payload,'stop':stop})
 check('Context-sensitive text stop',status==200 and response['choices'][0]['text']=='')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  results=list(pool.map(lambda _:req('/v1/completions',payload),range(4)))
 check('Concurrent CUDA sessions match isolated',all(status==200 and body['choices'][0]['text']==complete for status,body in results))
 metrics=req('/metrics')[1]
 check('Observable GPU launches',metrics['runtime']['gpu_kernel_launches']>0)
 check('Healthy CUDA execution',metrics['runtime']['backend_healthy'])
 check('No leaked active KV pages',metrics['scheduler']['kv_pages_used']==0)
 launches=metrics['runtime']['gpu_kernel_launches']
 check('Reload CUDA model',req('/v1/model/reload',{})[0]==200)
 check('Reload retains CUDA',req('/v1/model')[1]['backend']=='cuda')
 check('Load without backend preserves explicit CUDA',req('/v1/model/load',{'model':MODEL})[1]['backend']=='cuda')
 check('Generation after replacement',req('/v1/completions',payload)[1]['choices'][0]['text']==complete)
 check('Bad replacement rejected',req('/v1/model/load',{'model':'missing.gguf','backend':'cuda'})[0]==400)
 check('Bad replacement preserves serving model',req('/v1/completions',payload)[1]['choices'][0]['text']==complete)
 check('Unload CUDA model',req('/v1/model/unload',{})[0]==200)
 info=req('/v1/model')[1]
 check('Resident model weights released',info['device_weight_bytes']==0 and not info['loaded'])
 check('Generation fails after unload',req('/v1/completions',payload)[0]==503)
 print(json.dumps({'passed':len(checks),'checks':checks,'chat':text,'completion':complete,'gpu_kernel_launches_before_reload':launches},indent=2))
finally:
 proc.terminate();proc.wait(timeout=10);log.seek(0);print(log.read(),file=__import__('sys').stderr);log.close()
