import argparse, sys, time
p=argparse.ArgumentParser(); p.add_argument("--src",required=True); p.add_argument("--out",required=True); a=p.parse_args()
sys.path.insert(0,a.src)
import torch; torch.set_num_threads(1)
import lat
from lat.ir import LinearizedAttention,LinearizedAttribution,LinearizedLayer,LinearizedMLP,NormSnapshot
class Tok:
    def decode(self,ids):
        if isinstance(ids,int): return f"<t{ids}>"
        return " ".join(f"<t{int(i)}>" for i in torch.as_tensor(ids).flatten())
def norm(n_pos,d_model): return NormSnapshot(scale=torch.ones(n_pos),gamma=torch.ones(d_model),mean=None,beta=None)
def build(*,n_layers,n_pos,d_model,feats_per_layer,feat_positions,feat_layers=None,seed=0):
    g=torch.Generator().manual_seed(seed)
    bp=torch.tril(torch.ones(n_pos,n_pos)); bp=bp/bp.sum(-1,keepdim=True)
    layers=[]
    for i in range(n_layers):
        hf = feat_layers is None or i in feat_layers
        if hf:
            reps=max(1,feats_per_layer//len(feat_positions))
            positions=torch.tensor(sorted(feat_positions*reps),dtype=torch.long)
            na=positions.shape[0]; w=torch.randn(na,d_model,generator=g)*0.3
            mlp=LinearizedMLP(active_positions=positions,active_feature_idx=torch.arange(na,dtype=torch.long),active_activations=torch.ones(na),W_enc_active=w,W_dec_active=w*0.9,b_enc_active=torch.zeros(na),b_dec=torch.zeros(d_model),d_transcoder=max(na,64),pre_norm=norm(n_pos,d_model),error_vec=torch.zeros(n_pos,d_model))
        else:
            mlp=LinearizedMLP(active_positions=torch.zeros(0,dtype=torch.long),active_feature_idx=torch.zeros(0,dtype=torch.long),active_activations=torch.zeros(0),W_enc_active=torch.zeros(0,d_model),W_dec_active=torch.zeros(0,d_model),b_enc_active=torch.zeros(0),b_dec=torch.zeros(d_model),d_transcoder=64,pre_norm=norm(n_pos,d_model),error_vec=torch.zeros(n_pos,d_model))
        att=LinearizedAttention(pattern=bp.unsqueeze(0).repeat(2,1,1)*0.05,W_V=torch.randn(2,d_model,4,generator=g)*0.1,W_O=torch.randn(2,4,d_model,generator=g)*0.1,b_V=None,b_O=None,pre_norm=norm(n_pos,d_model),is_causal=True)
        layers.append(LinearizedLayer(layer_idx=i,attention=att,mlp=mlp))
    return LinearizedAttribution(input_tokens=torch.zeros(n_pos,dtype=torch.long),resid_0=torch.randn(n_pos,d_model,generator=g)*0.1,embed_vecs=torch.randn(n_pos,d_model,generator=g)*0.1,layers=tuple(layers),final_norm=norm(n_pos,d_model),W_unembed=torch.randn(d_model,32,generator=g),b_unembed=None,logits=torch.randn(n_pos,32,generator=g))
def trun(ir,reps=3):
    best=float("inf")
    for _ in range(reps):
        t0=time.process_time(); lat.attribute_from_linearized(ir,Tok(),show_progress=False,batch_size=16); best=min(best,time.process_time()-t0)
    return best
early=build(n_layers=4,n_pos=256,d_model=16,feats_per_layer=256,feat_positions=[1,2,3,4])
late=build(n_layers=4,n_pos=256,d_model=16,feats_per_layer=256,feat_positions=[252,253,254,255])
trun(early,1); te=trun(early); tl=trun(late)
low=build(n_layers=64,n_pos=8,d_model=16,feats_per_layer=128,feat_positions=list(range(8)),feat_layers=[0,1,2,3])
high=build(n_layers=64,n_pos=8,d_model=16,feats_per_layer=128,feat_positions=list(range(8)),feat_layers=[60,61,62,63])
trun(low,1); tlo=trun(low); thi=trun(high)
import json
json.dump({"temporal_asymmetry": tl/te, "layer_asymmetry": thi/tlo}, open(a.out,"w"))
