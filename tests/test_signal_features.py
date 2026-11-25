import numpy as np
from utils.signal_features import extract_signal_features, joint_probabilities_to_heads


def test_features_ignore_dc_offset_and_remain_finite_for_flat_channels():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(3,64,4)).astype(np.float32)
    x[:,:,0]=0
    for mode in ['center','center_detrend']:
        a=extract_signal_features(x,mode=mode)
        b=extract_signal_features(x+np.array([0,4,8,12],dtype=np.float32),mode=mode)
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a,b,rtol=3e-4,atol=3e-4)


def test_joint_marginals_preserve_action_and_active_finger_mass():
    a,f=joint_probabilities_to_heads(np.array([[.2,.3,.5],[1.,0.,0.]]),[0,1,6])
    np.testing.assert_allclose(a,[[.2,.3,.5],[1,0,0]])
    np.testing.assert_allclose(f,[[1,0,0,0,0],[.2,.2,.2,.2,.2]])
