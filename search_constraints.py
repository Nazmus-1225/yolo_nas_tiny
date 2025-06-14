search_space = {
    "stages": [3, 4, 5],
    "channel_sizes": [
        [32, 64, 128, 192, 256],
        [32, 48, 72, 108, 162],
        [48, 96, 192, 384, 512],
        [48, 72, 108, 162, 243],
        [64, 128, 256, 512, 1024],
        [64, 96, 144, 216, 324]    
    ],
    "c2f_repeats": [
        [1, 2, 2, 2],
        [2, 4, 4, 2],
        [3, 6, 6, 3]
    ],
    "include_sppf": [True, False]
}

constraints = {
    "max_model_size_kb": 900,     
    "max_params_m": 0.5,           
    "max_gflops": 3.0              
}
