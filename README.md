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
