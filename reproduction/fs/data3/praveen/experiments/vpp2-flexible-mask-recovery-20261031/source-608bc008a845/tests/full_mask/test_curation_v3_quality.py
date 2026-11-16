import numpy as np
import torch
from vpp2.full_mask.curation_v3_quality import ksg_cross_episode,SummaryVAE,reconstruction


def test_quality_distinguishes_consistent_from_scrambled_pairs():
    rng=np.random.default_rng(18);o=rng.normal(size=(160,3));a=o+.03*rng.normal(size=o.shape)
    ep=np.arange(160)//4;ref=np.arange(160)
    correct=ksg_cross_episode(o,a,ep,ref).mean()
    wrong=ksg_cross_episode(o,a[rng.permutation(160)],ep,ref).mean()
    assert correct>wrong+.5


def test_same_episode_copies_do_not_create_neighbours():
    rng=np.random.default_rng(3);o=rng.normal(size=(40,2));a=o+.1;ep=np.arange(40);ref=np.arange(40)
    first=ksg_cross_episode(o,a,ep,ref)
    # Extra queried copies from one episode are not in the independent reference pool.
    oo=np.concatenate([o,np.repeat(o[:1],20,0)]);aa=np.concatenate([a,np.repeat(a[:1],20,0)]);ee=np.r_[ep,np.zeros(20,dtype=int)]
    second=ksg_cross_episode(oo,aa,ee,ref)
    np.testing.assert_allclose(first,second[:40])


def test_small_scoring_model_is_not_a_policy_and_uses_deterministic_mean():
    torch.manual_seed(100008);model=SummaryVAE(40,8);x=torch.randn(6,40)
    assert torch.equal(model.encode(x),model.encode(x))
    p,mu,kl=model(x)
    loss=reconstruction(p,x,True)+.001*kl;loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
