# EDA-on-Titanic
This repo involved a comprehensive Exploratory Data Analysis (EDA) of the Titanic dataset to uncover patterns and insights related to passenger survival
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h" // Download this single file from the internet
#include <vector>
#include <iostream>
#include <string>

// Constants matching your Python script
const int SAMPLE_RATE = 22050;
const float DURATION = 1.25f;
const int AUDIO_LENGTH = static_cast<int>(SAMPLE_RATE * DURATION); // 27562

std::vector<float> load_audio_file(const std::string& file_path) {
    unsigned int channels;
    unsigned int file_sr;
    drwav_uint64 totalPCMFrameCount;

    // 1. LOAD AND NORMALIZE
    // This perfectly mimics librosa.load() + __soundfile_load()
    // It reads the file and automatically converts 16-bit integers to float32 (-1.0 to 1.0)
    float* pSampleData = drwav_open_file_and_read_pcm_frames_f32(
        file_path.c_str(), &channels, &file_sr, &totalPCMFrameCount, NULL);

    // Error handling (Matches your "except Exception as e:" block)
    if (pSampleData == NULL) {
        std::cerr << "Error loading " << file_path << std::endl;
        return std::vector<float>(AUDIO_LENGTH, 0.0f); // Return silent array
    }

    // Skipping Resampling logic as requested
    if (file_sr != SAMPLE_RATE) {
        std::cerr << "Warning: File is " << file_sr 
                  << "Hz, expected " << SAMPLE_RATE << "Hz. Skipping resampling.\n";
    }

    // 2. CONVERT TO MONO (Matches librosa's default "mono=True")
    std::vector<float> audio;
    if (channels > 1) {
        audio.reserve(totalPCMFrameCount);
        for (size_t i = 0; i < totalPCMFrameCount; ++i) {
            float sum = 0.0f;
            for (unsigned int c = 0; c < channels; ++c) {
                sum += pSampleData[i * channels + c];
            }
            audio.push_back(sum / channels); // Average the channels
        }
    } else {
        // Already mono, just copy the data
        audio.assign(pSampleData, pSampleData + totalPCMFrameCount);
    }

    // Free the raw buffer now that we have our vector
    drwav_free(pSampleData, NULL);

    // 3. PAD OR TRUNCATE
    // This one C++ line replaces your entire Python if/else length block!
    // If the vector is shorter than 27562, it pads the end with 0.0f.
    // If the vector is longer than 27562, it chops off the excess.
    audio.resize(AUDIO_LENGTH, 0.0f);

    return audio;
}



####################################################


#include <vector>
#include <string>
#include "librosa.h" // Ensure this is in your include path

// Constants matching your Python script
const int SAMPLE_RATE = 22050;
const int N_FFT = 2048;
const int HOP_LENGTH = 512;
const int N_MELS = 128;

std::vector<std::vector<float>> extract_mel_spectrogram(std::vector<float>& audio) {
    // 1. Explicitly define Python's hidden default parameters
    std::string win = "hann";
    bool center = true;
    std::string mode = "reflect";
    float power = 2.0f; // Librosa defaults to 2.0 (power spectrogram)
    int fmin = 0;
    int fmax = SAMPLE_RATE / 2; // Nyquist frequency

    // 2. Map standard C++ vector to Eigen Vectorf (Zero-copy mapping)
    librosa::Vectorf map_x = Eigen::Map<librosa::Vectorf>(audio.data(), audio.size());

    // 3. Calculate Mel Spectrogram
    // We transpose it here so the shape becomes (Frames, Mels), 
    // which is the standard format required for TFLite 1D/2D Convolutions.
    librosa::Matrixf mel_spec = librosa::internal::melspectrogram(
        map_x, SAMPLE_RATE, N_FFT, HOP_LENGTH, win, center, mode, power, N_MELS, fmin, fmax
    ).transpose();

    // 4. Convert to decibels (log scale)
    // internal::power2db flawlessly replicates `librosa.power_to_db(mel_spec, ref=np.max)`.
    // It finds the max coefficient dynamically and clamps the floor to -80 decibels.
    librosa::Matrixf log_mel_spec_matrix = librosa::internal::power2db(mel_spec);

    // 5. Map the Eigen Matrix back to a standard C++ 2D vector for your ML pipeline
    std::vector<std::vector<float>> log_mel_spec(
        log_mel_spec_matrix.rows(), 
        std::vector<float>(log_mel_spec_matrix.cols(), 0.f)
    );
    
    for (int i = 0; i < log_mel_spec_matrix.rows(); ++i) {
        auto &row = log_mel_spec[i];
        Eigen::Map<librosa::Vectorf>(row.data(), row.size()) = log_mel_spec_matrix.row(i);
    }

    return log_mel_spec;
}

#############################################################

#include <vector>
#include <string>
#include <iostream>

// (Assuming load_audio_file and extract_mel_spectrogram are defined above this)

// We use a 3D vector to represent the [Batch, Mels, Time] shape.
// Since Batch is 1, the outermost vector will just have a size of 1.
std::vector<std::vector<std::vector<float>>> preprocess_audio(const std::string& file_path) {
    try {
        // 1. Load the audio (Returns a 1D vector of 27,562 floats)
        std::vector<float> audio = load_audio_file(file_path);

        // Check if loading failed (our load_audio_file returns an empty/silent vector on fail)
        if (audio.empty()) {
            throw std::runtime_error("Audio loading failed or returned empty.");
        }

        // 2. Extract features (Returns a 2D vector matrix of Decibels)
        std::vector<std::vector<float>> features = extract_mel_spectrogram(audio);

        // 3. Add batch dimension (Equivalent to np.expand_dims(features, axis=0))
        // This takes our 2D matrix and puts it inside a 3D array structure.
        std::vector<std::vector<std::vector<float>>> batched_features;
        batched_features.push_back(features);

        return batched_features;

    } catch (const std::exception& e) {
        std::cerr << "Error preprocessing " << file_path << ": " << e.what() << std::endl;
        
        // Return an empty 3D vector to represent "None"
        return std::vector<std::vector<std::vector<float>>>(); 
    }
}




##############################################

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <iostream>
#include <vector>
#include <cstring> // for std::memcpy

// Assume 'flat_features' is your 2D Mel Spectrogram flattened into a 1D vector 
// (using the helper function we discussed previously)
int run_tflite_inference(const std::string& model_path, const std::vector<float>& flat_features) {
    
    // 1. Load the Model from disk
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        
    if (!model) {
        std::cerr << "Failed to load TFLite model\n";
        return -1;
    }

    // 2. Build the Interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter) {
        std::cerr << "Failed to construct interpreter\n";
        return -1;
    }

    // 3. Allocate Tensors (EXACT MATCH to interpreter.allocate_tensors())
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!\n";
        return -1;
    }

    // 4. Set Input Tensor (EXACT MATCH to interpreter.set_tensor)
    // Get the pointer to the memory address of the model's input
    float* input_tensor_ptr = interpreter->typed_input_tensor<float>(0);
    
    // Get the expected size of the input tensor to prevent memory corruption
    TfLiteTensor* raw_input_tensor = interpreter->input_tensor(0);
    int expected_bytes = raw_input_tensor->bytes;
    int provided_bytes = flat_features.size() * sizeof(float);

    if (expected_bytes != provided_bytes) {
        std::cerr << "SIZE MISMATCH! Model expects " << expected_bytes 
                  << " bytes, but you provided " << provided_bytes << " bytes.\n";
        return -1;
    }

    // Copy the feature data directly into the model's memory space
    std::memcpy(input_tensor_ptr, flat_features.data(), expected_bytes);

    // 5. Invoke the Model (EXACT MATCH to interpreter.invoke())
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke tflite!\n";
        return -1;
    }

    // 6. Get Output Tensor (EXACT MATCH to interpreter.get_tensor)
    float* output_tensor_ptr = interpreter->typed_output_tensor<float>(0);
    TfLiteTensor* raw_output_tensor = interpreter->output_tensor(0);
    int num_classes = raw_output_tensor->bytes / sizeof(float);

    // 7. Find ArgMax (EXACT MATCH to np.argmax)
    int best_class = 0;
    float max_prob = output_tensor_ptr[0];

    for (int i = 1; i < num_classes; ++i) {
        if (output_tensor_ptr[i] > max_prob) {
            max_prob = output_tensor_ptr[i];
            best_class = i;
        }
    }

    return best_class;
}
##############################################################

https://github.com/mackron/dr_libs/blob/master/dr_wav.h
https://github.com/ewan-xu/LibrosaCpp/blob/main/librosa/librosa.h
curl -L -O https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
mkdir eigen3
mv eigen-3.4.0/Eigen eigen3/


#######################################################################################################################################
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


# =========================================================
# SAFE PARAMETER RANGES
# These ranges are intentionally chosen so that:
# 1. anomalies are clearly distinguishable from normal
# 2. anomalies are NOT absurdly unrealistic
# =========================================================

PARAMS = {
    "harmonic_shift": {
        "freq_shift_bins": (6, 18),      # IMPORTANT
        "region_h": (24, 72),
        "region_w": (18, 64),
    },

    "spectral_warp": {
        "warp_strength": (0.18, 0.45),   # IMPORTANT
        "region_h": (32, 96),
        "region_w": (24, 80),
    },

    "frequency_smearing": {
        "blur_sigma": (2.0, 5.0),        # IMPORTANT
        "region_h": (32, 96),
        "region_w": (18, 72),
    },

    "bandwidth_degradation": {
        "downsample_factor": (2, 5),     # IMPORTANT
        "region_h": (48, 128),
        "region_w": (32, 96),
    }
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def random_region(spec, h_range, w_range):
    H, W = spec.shape

    rh = np.random.randint(h_range[0], h_range[1])
    rw = np.random.randint(w_range[0], w_range[1])

    y1 = np.random.randint(0, H - rh)
    x1 = np.random.randint(0, W - rw)

    return y1, y1 + rh, x1, x1 + rw


def gaussian_mask(h, w, sigma_ratio=0.25):

    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)

    xx, yy = np.meshgrid(x, y)

    sigma = sigma_ratio

    mask = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    mask = mask / mask.max()

    return mask


def blend(original, modified, mask):
    return original * (1 - mask) + modified * mask


# =========================================================
# 1. HARMONIC SHIFT
# =========================================================

def harmonic_shift(spec):

    spec = spec.copy()

    p = PARAMS["harmonic_shift"]

    y1, y2, x1, x2 = random_region(
        spec,
        p["region_h"],
        p["region_w"]
    )

    shift = np.random.randint(
        p["freq_shift_bins"][0],
        p["freq_shift_bins"][1]
    )

    region = spec[y1:y2, x1:x2]

    shifted = np.roll(region, shift, axis=0)

    mask = gaussian_mask(y2 - y1, x2 - x1)

    spec[y1:y2, x1:x2] = blend(
        region,
        shifted,
        mask
    )

    return spec


# =========================================================
# 2. SPECTRAL WARP
# =========================================================

def spectral_warp(spec):

    spec = spec.copy()

    p = PARAMS["spectral_warp"]

    y1, y2, x1, x2 = random_region(
        spec,
        p["region_h"],
        p["region_w"]
    )

    region = spec[y1:y2, x1:x2]

    h, w = region.shape

    strength = np.random.uniform(
        p["warp_strength"][0],
        p["warp_strength"][1]
    )

    warped = np.zeros_like(region)

    for x in range(w):

        shift = int(
            strength * np.sin(2 * np.pi * x / w) * h * 0.25
        )

        warped[:, x] = np.roll(region[:, x], shift)

    mask = gaussian_mask(h, w)

    spec[y1:y2, x1:x2] = blend(
        region,
        warped,
        mask
    )

    return spec


# =========================================================
# 3. FREQUENCY SMEARING
# =========================================================

def frequency_smearing(spec):

    spec = spec.copy()

    p = PARAMS["frequency_smearing"]

    y1, y2, x1, x2 = random_region(
        spec,
        p["region_h"],
        p["region_w"]
    )

    region = spec[y1:y2, x1:x2]

    sigma = np.random.uniform(
        p["blur_sigma"][0],
        p["blur_sigma"][1]
    )

    # IMPORTANT:
    # vertical blur >> horizontal blur
    smeared = gaussian_filter(
        region,
        sigma=(sigma, 0.6)
    )

    mask = gaussian_mask(y2 - y1, x2 - x1)

    spec[y1:y2, x1:x2] = blend(
        region,
        smeared,
        mask
    )

    return spec


# =========================================================
# 4. BANDWIDTH DEGRADATION
# =========================================================

def bandwidth_degradation(spec):

    spec = spec.copy()

    p = PARAMS["bandwidth_degradation"]

    y1, y2, x1, x2 = random_region(
        spec,
        p["region_h"],
        p["region_w"]
    )

    region = spec[y1:y2, x1:x2]

    h, w = region.shape

    factor = np.random.randint(
        p["downsample_factor"][0],
        p["downsample_factor"][1]
    )

    # downsample
    small = cv2.resize(
        region,
        (max(1, w // factor), max(1, h // factor)),
        interpolation=cv2.INTER_LINEAR
    )

    # upsample back
    degraded = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    mask = gaussian_mask(h, w)

    spec[y1:y2, x1:x2] = blend(
        region,
        degraded,
        mask
    )

    return spec


# =========================================================
# MAIN GENERATOR
# =========================================================

DISTORTION_FUNCS = {
    "harmonic_shift": harmonic_shift,
    "spectral_warp": spectral_warp,
    "frequency_smearing": frequency_smearing,
    "bandwidth_degradation": bandwidth_degradation,
}


def generate_frequency_distortion(spec):

    distortion_type = np.random.choice(
        [
            "harmonic_shift",
            "spectral_warp",
            "frequency_smearing",
            "bandwidth_degradation"
        ],
        p=[0.30, 0.30, 0.25, 0.15]
    )

    distorted = DISTORTION_FUNCS[distortion_type](spec)

    return distorted, distortion_type
