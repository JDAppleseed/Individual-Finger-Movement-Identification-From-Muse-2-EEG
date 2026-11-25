from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from utils.subject_ensemble import SubjectEnsemblePredictor


def predictor():
    p = SubjectEnsemblePredictor.__new__(SubjectEnsemblePredictor)
    p.spec = {'input_shape': [64,4], 'fs': 256., 'feature_mode': 'center'}
    p.weight = .5
    p.scaler = p.model = p.temperature = None
    p.device = 'cpu'
    p.feature_model = SimpleNamespace(classes_=np.array([0,1,6]),
        predict_proba=lambda x: np.array([[.2,.6,.2]]))
    return p


def cnn_result():
    return dict(action_probs=np.array([.1,.2,.7]), finger_probs=np.ones(5)/5,
                action_logits=np.ones(3), finger_logits=np.ones(5),
                finger_applicable_prob=.37, applicability_logit=-.53)


def test_runtime_blends_heads_but_preserves_cnn_applicability():
    p=predictor();x=np.ones((64,4),dtype=np.float32)
    with patch('utils.live_infer_common.predict_window',side_effect=lambda *a,**k:cnn_result()), \
         patch('utils.subject_ensemble.apply_channel_normalizer',side_effect=lambda x,s:x):
        out=p.predict_window(x,prepared_window=x.copy())
    np.testing.assert_allclose(out['action_probs'],[.15,.4,.45])
    np.testing.assert_allclose(out['finger_probs'],[.6,.1,.1,.1,.1])
    assert out['finger_applicable_prob']==.37
    assert out['action_logits'] is None
    assert out['backend']=='subject_ensemble'


def test_quality_modified_input_falls_back_without_calling_feature_model():
    p=predictor();x=np.ones((64,4),dtype=np.float32)
    p.feature_model.predict_proba=lambda x:pytest.fail('Feature model must not see modified-quality windows')
    with patch('utils.live_infer_common.predict_window',side_effect=lambda *a,**k:cnn_result()) as call, \
         patch('utils.subject_ensemble.apply_channel_normalizer',side_effect=lambda x,s:x):
        prepared=x.copy();prepared[:,0]=0
        out=p.predict_window(x,prepared_window=prepared)
    np.testing.assert_array_equal(out['action_probs'],[.1,.2,.7])
    np.testing.assert_array_equal(call.call_args.kwargs['prepared_window'],prepared)
    assert out['ensemble_fallback']=='quality_modified_input'


@pytest.mark.parametrize('x',[np.zeros((63,4)),np.full((64,4),np.nan),np.full((64,4),np.inf)])
def test_invalid_windows_are_rejected(x):
    with pytest.raises(ValueError,match='Invalid raw ensemble'):
        predictor().predict_window(x)
