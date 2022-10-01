CREATE TABLE IF NOT EXISTS models (
    model_id INT AUTO_INCREMENT PRIMARY KEY,
    model VARCHAR(64),
    params VARCHAR(2048),
    d INT,
    n_classes INT,
    n_trained INT,
    model_pkl BLOB
);
