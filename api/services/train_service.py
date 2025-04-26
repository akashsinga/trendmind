# api/services/train_service.py

from core.trainer.ensemble_trainer import run_ensemble_training

def run_ensemble_training_service():
  try:
    run_ensemble_training()
    return {'status': 'success', 'message': 'Ensemble model training completed'}
  except Exception as e:
    return {'status': 'error', 'message': str(e)}