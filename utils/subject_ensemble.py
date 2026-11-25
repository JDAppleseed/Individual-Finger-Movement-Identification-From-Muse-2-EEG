"""Inference for offline CNN/signal-feature ensembles saved by the subject benchmark."""
from pathlib import Path
import json
import joblib
import numpy as np
import torch
from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.model_outputs import infer_output_dims_from_state_dict, unpack_model_outputs
from utils.runtime_utils import load_normalizer, apply_channel_normalizer, load_temperature_scaling
from utils.signal_features import extract_signal_features, joint_probabilities_to_heads


def predict_subject_ensemble(X, manifest_path, *, device='cpu', batch_size=256):
    manifest_path=Path(manifest_path)
    spec=json.loads(manifest_path.read_text())
    X=np.asarray(X,dtype=np.float32)
    if X.ndim!=3 or not len(X) or not np.isfinite(X).all():
        raise ValueError('Expected nonempty finite [windows,time,channels] input')
    if 'input_shape' in spec and list(X.shape[1:])!=spec['input_shape']:
        raise ValueError('Window shape does not match the saved model')
    weight=float(spec['cnn_weight'])
    if not 0<=weight<=1:
        raise ValueError('cnn_weight must be between zero and one')
    action=np.zeros((len(X),3),dtype=np.float32)
    finger=np.zeros((len(X),5),dtype=np.float32)
    if weight<1:
        feature_model=joblib.load(manifest_path.parent/spec['feature_model'])
        features=extract_signal_features(X,fs=spec['fs'],mode=spec['feature_mode'])
        a,f=joint_probabilities_to_heads(feature_model.predict_proba(features),feature_model.classes_)
        action+=(1-weight)*a;finger+=(1-weight)*f
    if weight>0:
        root=Path(__file__).resolve().parents[1]
        run=root/spec['cnn_run']
        state=torch.load(run/'finger_action_model.pt',map_location='cpu',weights_only=True)
        nf,na,has_app=infer_output_dims_from_state_dict(state)
        if nf!=5 or na!=3:
            raise ValueError('Ensemble expects the active five-finger and three-action heads')
        model=CNNLSTMFingerActionNet(n_channels=X.shape[2],n_fingers=nf,n_actions=na,finger_applicability_head=has_app)
        model.load_state_dict(state);model.to(device);model.eval()
        normalizer=load_normalizer(run/'scaler.npz')
        if normalizer is None:raise ValueError('Missing normalizer')
        temp=load_temperature_scaling(run/'temperature_scaling.json')
        with torch.inference_mode():
            for start in range(0,len(X),batch_size):
                end=min(start+batch_size,len(X))
                batch=torch.as_tensor(apply_channel_normalizer(X[start:end],normalizer),device=device)
                fl,al,_=unpack_model_outputs(model(batch))
                action[start:end]+=weight*torch.softmax(al/temp.action_temperature,dim=1).cpu().numpy()
                finger[start:end]+=weight*torch.softmax(fl/temp.finger_temperature,dim=1).cpu().numpy()
    return action,finger


class SubjectEnsemblePredictor:
    """Persistent deterministic predictor for replay and live shadow inference.

    The CNN retains the applicability head. If quality processing modifies its
    normalized input, use the quality-processed CNN alone for that window: the
    feature classifier was trained on intact raw windows, not masked channels.
    This fallback is observable and must be evaluated with the command policy.
    """

    def __init__(self, manifest_path, *, device='cpu'):
        from utils.live_infer_common import load_model_artifacts
        self.manifest_path = Path(manifest_path).resolve()
        self.spec = json.loads(self.manifest_path.read_text())
        self.weight = float(self.spec['cnn_weight'])
        if not 0 < self.weight <= 1:
            raise ValueError('Live ensembles require a CNN applicability head')
        self.device = torch.device(device)
        self.run = Path(__file__).resolve().parents[1] / self.spec['cnn_run']
        self.model, self.scaler, self.temperature = load_model_artifacts(
            run_dir=self.run, device=self.device,
            n_channels=int(self.spec['input_shape'][-1]))
        if self.model.finger_applicability_head is None:
            raise ValueError('Live ensembles require an applicability head')
        self.feature_model = None
        if self.weight < 1:
            self.feature_model = joblib.load(self.manifest_path.parent / self.spec['feature_model'])
            if hasattr(self.feature_model, 'n_jobs'):
                self.feature_model.set_params(n_jobs=1)

    def predict_window(self, window, *, prepared_window=None):
        from utils.live_infer_common import predict_window
        window = np.asarray(window, dtype=np.float32)
        if list(window.shape) != self.spec['input_shape'] or not np.isfinite(window).all():
            raise ValueError('Invalid raw ensemble window shape or values')
        result = predict_window(window, scaler=self.scaler, model=self.model,
                                device=self.device, inference_engine=None,
                                temperature_state=self.temperature,
                                prepared_window=prepared_window)
        result['backend'] = 'subject_ensemble'
        result['ensemble_fallback'] = None
        normalized = apply_channel_normalizer(window, self.scaler)
        if prepared_window is not None and not np.array_equal(normalized, prepared_window):
            result['backend'] = 'subject_ensemble_quality_cnn_fallback'
            result['ensemble_fallback'] = 'quality_modified_input'
            return result
        if self.feature_model is not None:
            features = extract_signal_features(window[None], fs=self.spec['fs'],
                                               mode=self.spec['feature_mode'])
            a, f = joint_probabilities_to_heads(self.feature_model.predict_proba(features),
                                                self.feature_model.classes_)
            result['action_probs'] = self.weight * result['action_probs'] + (1-self.weight) * a[0]
            result['finger_probs'] = self.weight * result['finger_probs'] + (1-self.weight) * f[0]
            # No single neural logit vector represents this mixture.
            result['action_logits'] = None
            result['finger_logits'] = None
        return result
