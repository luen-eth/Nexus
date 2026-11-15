import struct,pathlib,json,sys
root=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/nexus-rope-reference');root.mkdir(parents=True,exist_ok=True)
H,V,FF=8,8,16
specs=[('token_embd.weight',V,H),('output.weight',V,H),('blk.0.attn_q.weight',H,H),('blk.0.attn_k.weight',4,H),('blk.0.attn_v.weight',4,H),('blk.0.attn_output.weight',H,H),('blk.0.ffn_gate.weight',FF,H),('blk.0.ffn_up.weight',FF,H),('blk.0.ffn_down.weight',H,FF)]
tensors=[]
for salt,(name,rows,cols) in enumerate(specs,1):
 values=[struct.unpack('<f',struct.pack('<f',((i*17+salt*7)%31)/31-.45))[0] for i in range(rows*cols)]
 tensors.append((name,[cols,rows],values))
for name in ['output_norm.weight','blk.0.attn_norm.weight','blk.0.ffn_norm.weight']:tensors.append((name,[H],[1.]*H))
def string(s):
 b=s.encode();return struct.pack('<Q',len(b))+b
meta={
 'general.architecture':(8,string('llama')),'general.name':(8,string('nexus-rope-reference')),
 'tokenizer.ggml.model':(8,string('llama')),
 'tokenizer.ggml.tokens':(9,struct.pack('<IQ',8,8)+b''.join(string(x) for x in ['<unk>','<s>','</s>','a','b','c','d','e'])),
 'tokenizer.ggml.scores':(9,struct.pack('<IQ',6,8)+struct.pack('<8f',*([0.]*8))),
 'tokenizer.ggml.token_type':(9,struct.pack('<IQ',5,8)+struct.pack('<8i',2,3,3,1,1,1,1,1)),
 'tokenizer.ggml.add_bos_token':(7,b'\0'),'tokenizer.ggml.add_space_prefix':(7,b'\0'),
}
for k,v in {'block_count':1,'embedding_length':H,'feed_forward_length':FF,'attention.head_count':2,'attention.head_count_kv':1,'context_length':32,'vocab_size':V,'rope.dimension_count':4}.items():meta['llama.'+k]=(4,struct.pack('<I',v))
for k,v in {'rope.freq_base':10000.,'attention.layer_norm_rms_epsilon':1e-5}.items():meta['llama.'+k]=(6,struct.pack('<f',v))
data=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(meta))
for k,(t,v) in sorted(meta.items()):data+=string(k)+struct.pack('<I',t)+v
payload=b''
for name,shape,values in tensors:
 data+=string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)+struct.pack('<IQ',0,len(payload))
 payload+=struct.pack('<'+'f'*len(values),*values);payload+=bytes((-len(payload))%32)
data+=bytes((-len(data))%32);(root/'tiny.gguf').write_bytes(data+payload)
(root/'weights.json').write_text(json.dumps(tensors))
