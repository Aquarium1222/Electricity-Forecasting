import sys
import numpy as np

data = [2790, 3080, 3384, 3636, 2695, 3009, 3104, 3235, 3131, 3298, 3851, 3091, 3157, 3412, 3386, 3005, 3399, 3160,
        3334, 3183, 3259, 3887, 3344, 3483, 3551, 3528, 3704, 3653, 3304, 2991, 3361, 3890, 3247, 4088, 3302, 3643,
        3347, 3767, 3660, 3084, 3146, 3219, 3348, 3246, 3123, 2996, 3985, 3744, 4249, 3564, 3132, 3138, 3674, 3739,
        3010, 3124, 3167, 4253, 3286, 2806, 3321, 3295, 3106, 3758, 3518, 3179, 3712, 3379, 3209, 3931, 3255, 3002,
        3098, 3826, 3493, 4820, 4143, 4153, 3085, 3700, 3020, 3086, 3771, 4125, 3577, 3187, 3264, 4066, 3489, 3508,
        3322, 4095, 4108, 3749, 4414, 4249, 4054, 4037, 3364, 3678, 4650, 3179, 2809, 3328, 3563, 3497, 3144, 3609,
        3489, 3028, 4091, 4376, 4031, 3948, 3948, 3166, 3120, 3810, 4687, 3346, 4864, 3512, 3414, 2967, 3476, 3400,
        3426, 3736, 3414, 3823, 3015, 3350, 3766, 4199, 3382, 3622, 3947, 4241, 4085, 4626, 4951, 3983, 3675, 4061,
        4289, 3942, 3563, 4932, 5261, 4098, 4186, 3430, 3633, 4071, 4163, 3754, 3682, 4240, 4535, 4580, 4515, 4367,
        3986, 3586, 3764, 4572, 5033, 4022, 3975, 4191, 3862, 3675, 3484, 3849, 3768, 4086, 3379, 3907, 3620, 3412,
        4013, 4646, 4029, 4458, 4131, 4287, 3475, 5016, 4610, 4315, 4367, 4379, 3671, 3518, 3974, 3924, 4210, 5085,
        4256, 4321, 4128, 4268, 5133, 4266, 3848, 4129, 4697, 4718, 4565, 4530, 4845, 5460, 4138, 4121, 3810, 4898,
        4353, 4387, 3853, 3860, 3871, 3187, 3726, 4384, 4270, 4651, 5022, 4941, 4622, 4585, 4236, 3869, 4384, 3862,
        3398, 3297, 3695, 3812, 3796, 4545, 4272, 4174, 4086, 4719, 4478, 3868, 4811, 4870, 3922, 3873, 4362, 4155,
        4176, 4255, 4084, 3383, 3024, 3718, 3725, 3733, 3724, 3729, 3317, 3885, 4562, 5125, 4772, 5434, 5274, 4547,
        4489, 4555, 4259, 5285, 4645, 3460, 3418, 3591, 4539, 4239, 4100, 4718, 4310, 3295, 3088, 3361, 3465, 3992,
        3926, 3801, 4087, 3946, 4505, 4244, 3309, 4341, 4003, 3892, 4025, 3503, 3402, 3470, 3707, 4215, 3719, 3569,
        3523, 3189, 3551, 3319, 3263, 3199, 3928, 4080, 3590, 3708, 3190, 3373, 3364, 3463, 3147, 3202, 3361, 3314,
        3643, 2928, 2765, 3144, 3134, 3306, 3144, 3495, 2977, 3287, 3320, 3237, 3050, 3168, 3084, 3618, 2691, 3106,
        3164, 3094, 3086, 3129, 3282, 2830, 3022, 3012, 3020, 3000, 3047, 3047, 3616, 3072, 3199, 3076, 3545, 3418,
        3066, 2869, 3135, 3094, 3118, 3154, 2767, 2936, 3037, 3032, 3067, 3102, 3102, 3106, 3201, 2906, 3093, 3116,
        3114, 3132, 3309, 3046, 3034, 3157, 3158, 3150, 3074, 3097, 3203, 3073, 3269, 3219, 3064, 3286, 3203, 3447,
        2906, 2938, 3046, 3006, 3132, 3060, 3066, 2763, 3047, 2880, 2887, 2833, 2992, 3024, 2911, 2644, 3001, 2824,
        2907, 2885, 2975, 3139, 2989, 3109, 3049, 3028, 3041, 2758, 2672, 2832, 3221, 3075, 3088, 3268, 2805, 2751,
        3110, 3048, 3055, 3076, 3272, 3075, 2933, 3075, 3087, 3183, 3214, 3137, 2838, 3186, 3225, 3537, 3059, 3272,
        3279, 3201, 2919, 3181, 3262, 3229, 3470, 3160, 3057, 2995, 2872, 3201, 3247, 3298, 3211, 2966, 2848, 3205,
        3195, 3235, 3208, 3320, 2883, 2863, 3091, 3214, 3302, 3285, 3334, 2989, 2911, 3197, 3474, 3169, 3296, 3058,
        2884, 2975, 3234, 3360, 3578, 3465, 3518, 3305, 2975, 3519, 3611, 3669, 3682, 3344, 3507, 3306, 1437, 3095,
        2880, 2570, 2882, 3653, 2244, 2285, 2797, 2388, 2523, 2326, 4990, 3604, 4077, 3562, 3989, 3799, 3726, 3987,
        4063, 4602, 3775, 3914, 3749, 2900, 3046, 2729, 3273, 3750, 3715, 3833, 3848, 4685, 4156, 3986, 3589, 3901,
        4694, 5589, 4203, 4320, 5005, 5282, 5987, 6038, 4602, 4326, 4525, 2268, 3924, 3799, 3805, 3858, 3811, 3394,
        3464, 2872, 3832, 4013, 4235, 4113, 4457, 3338, 4042, 5476, 5400, 5657, 4373, 3494, 4518, 3950, 3033, 3834,
        3833, 5201, 5486, 4516, 5591, 4495, 4526, 4637, 3438, 4066, 4845, 4897, 5975, 5950, 5481, 5364, 3483, 4420,
        4393, 4364, 4913, 5119, 4736, 3853, 4656, 3983, 3951, 4487, 4788, 5328, 4682, 3749, 4829, 3849, 4267, 3969,
        5140, 4562, 3883, 3843, 3860, 3902, 3303, 3720, 4008, 3846, 3958, 3899, 4841, 4647, 4609, 3153, 6911, 6556,
        3816, 3133, 4457, 5348, 3209, 4395, 5268, 4843, 5212, 3175, 5235, 3317, 3756, 2670, 2327, 1629, 1854, 5140,
        3642, 2614, 2384, 2572, 2884, 2808, 5663, 6547, 5902, 5502, 5030, 6040, 3899, 4684, 5049, 4043, 3471, 3915,
        6817, 5412, 3806, 4487, 3357, 2949, 4206, 3767, 2818, 5471, 2584, 5417, 3914, 3686, 3857, 4318, 5117, 4827,
        3571, 3899, 3859, 3466, 5704, 4094, 4098, 6608, 4036, 3858, 3937, 4089, 3727, 3435, 2773, 4007, 4556, 3821,
        4966, 4418, 3620, 3792, 2841, 4244, 3256, 3531, 3273, 4515, 4083, 3050, 2686, 2864, 6616, 4279, 3428, 2642,
        2862, 3057, 2423, 3191, 3035, 3429, 3767, 3252, 2807, 4376, 3927, 3346, 3641, 3184, 2620, 2512, 3209, 3010,
        3147, 3158, 2491, 2414, 2220, 2475, 2681, 2293, 3429, 2716, 2425, 3002, 2661, 2179, 3032, 2301, 3167, 2963,
        2549, 3464, 4700, 3346, 3087, 3111, 4421, 3233, 3136, 3577, 2931, 3108, 2986, 3395, 2833, 2972, 3343, 2921,
        3757, 2966, 2877, 3665, 3043, 3184, 2989, 2972, 2006, 2603, 2567, 2511, 3406, 3437, 3730, 4127]

train_data = []
for i in range(0, int(len(data) * 0.8)):
    train_data.append(data[i])

test_data = []
for i in range(int(len(data) * 0.8), len(data)):
    test_data.append(data[i])

min = int(sys.float_info.min)
min_poly = 0
for poly in range(0, 15):
    poly += 1

    print('poly: ' + str(poly), end = '\t\t\t')
    x = []
    for i in range(0, len(train_data)):
        x.append(i + 1)

    z = np.polyfit(x, train_data, poly)
    p = np.poly1d(z)

    RMSE = 0.0
    for x in range(633, 791):
        RMSE += pow(p(x) - test_data[x - 633], 2)

    RMSE = RMSE / len(train_data)
    RMSE = pow(RMSE, 0.5)

    if RMSE < min:
        min = RMSE
        min_poly = poly
    print('RMSE: ' + str(int(RMSE)), end = '\n')
