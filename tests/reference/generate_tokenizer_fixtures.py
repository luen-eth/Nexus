from tokenizers import Tokenizer,models,normalizers,pre_tokenizers,processors,decoders,trainers,AddedToken
from pathlib import Path
import json,jinja2
root=Path(__file__).resolve().parent.parent/'fixtures/tokenizers'
corpus=['',' hello','  hello world!','Türkçe İSTANBUL ıi','café cafe\u0301','中文 你好','🙂🚀','a\n\nb\tc','<s>hello</s>','hello END again','hello '*80]
wp=Tokenizer(models.WordPiece(vocab={'[UNK]':0,'[CLS]':1,'[SEP]':2,'hello':3,'world':4,'!':5,'cafe':6,'a':7,'b':8,'c':9,'##s':10},unk_token='[UNK]'))
wp.normalizer=normalizers.Sequence([normalizers.NFD(),normalizers.Lowercase(),normalizers.StripAccents()]);wp.pre_tokenizer=pre_tokenizers.BertPreTokenizer();wp.decoder=decoders.WordPiece(prefix='##',cleanup=True)
wp.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',special_tokens=[('[CLS]',1),('[SEP]',2)]);wp.add_special_tokens(['[UNK]','[CLS]','[SEP]'])
bpe=Tokenizer(models.BPE(unk_token=None));bpe.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);bpe.decoder=decoders.ByteLevel()
bpe.train_from_iterator(corpus,trainers.BpeTrainer(vocab_size=320,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['<s>','</s>','<|eot_id|>']))
bpe.post_processor=processors.TemplateProcessing(single='<s> $A',special_tokens=[('<s>',bpe.token_to_id('<s>'))])
vocab=[('<unk>',0.),('<s>',0.),('</s>',0.),('▁',-1.),('▁hello',4.),('▁world',4.),('!',1.),('a',-2.),('b',-2.),('c',-2.)]+[(f'<0x{i:02X}>',-10.) for i in range(256)]
uni=Tokenizer(models.Unigram(vocab,unk_id=0,byte_fallback=True));uni.normalizer=normalizers.NFKC();uni.pre_tokenizer=pre_tokenizers.Metaspace(replacement='▁',prepend_scheme='always');uni.decoder=decoders.Sequence([decoders.Replace('▁',' '),decoders.ByteFallback(),decoders.Fuse(),decoders.Strip(' ',1,0)])
uni.add_special_tokens(['<unk>','<s>','</s>']);uni.post_processor=processors.TemplateProcessing(single='<s> $A </s>',special_tokens=[('<s>',1),('</s>',2)])
template="{{ bos_token }}{% for message in messages %}{{ message['role'].upper() + ':' + message['content'].strip() + eos_token }}{% endfor %}{% if add_generation_prompt %}ASSISTANT:{% endif %}"
messages=[{'role':'system','content':' Be concise. '},{'role':'user','content':' Türkçe 🙂 '},{'role':'assistant','content':'Tamam.'},{'role':'user','content':'你好'}]
all_results=[]
for name,t,bos,eos in [('wordpiece',wp,'[CLS]','[SEP]'),('byte-bpe',bpe,'<s>','</s>'),('unigram',uni,'<s>','</s>')]:
 t.save(str(root/(name+'.json')))
 cfg={'bos_token':bos,'eos_token':eos,'chat_template':template}
 rendered=jinja2.Environment(trim_blocks=True,lstrip_blocks=True).from_string(template).render(messages=messages,add_generation_prompt=True,**cfg)
 cases=[]
 for text in corpus:
  cases.append({'text':text,'ids':t.encode(text,add_special_tokens=True).ids,'fragment_ids':t.encode(text,add_special_tokens=False).ids,'decoded':t.decode(t.encode(text).ids,skip_special_tokens=True)})
 all_results.append({'name':name,'vocab_size':max(t.get_vocab().values())+1,'config':cfg,'cases':cases,'messages':messages,'rendered':rendered,'chat_ids':t.encode(rendered,add_special_tokens=False).ids})
(root/'golden.json').write_text(json.dumps({'source':'Python tokenizers 0.22.2 + Jinja2','families':all_results},ensure_ascii=False,indent=2))

# HF overrides the default Jinja HTML-safe tojson filter.
env=jinja2.Environment(trim_blocks=True,lstrip_blocks=True)
env.filters['tojson']=lambda value,ensure_ascii=False,indent=None,separators=None,sort_keys=False:json.dumps(value,ensure_ascii=ensure_ascii,indent=indent,separators=separators,sort_keys=sort_keys)
messages=[{'role':'user','content':"<>&' Türkçe 🙂"}]
templates=['{{ messages | tojson }}','{{ messages | tojson(ensure_ascii=true) }}','{{ messages | tojson(indent=2,sort_keys=true) }}',"{{ messages | tojson(separators=(',', ':')) }}","{{ messages | tojson(false, '\t', (',', ': '), true) }}"]
(root/'json-template-golden.json').write_text(json.dumps([{'template':t,'rendered':env.from_string(t).render(messages=messages)} for t in templates],ensure_ascii=False,indent=2))

bare=Tokenizer(models.WordLevel(vocab={'x':0,'<0xC3>':1,'<0xA9>':2,'<unk>':3},unk_token='<unk>'))
bare.decoder=decoders.ByteFallback()
bare.save(str(root/'bare-byte-fallback.json'))
