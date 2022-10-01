UPDATE models
SET model_pkl = %s, n_trained = %s
WHERE model_id = %s;
