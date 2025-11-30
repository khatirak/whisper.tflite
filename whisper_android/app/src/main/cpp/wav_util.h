#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>

#pragma pack(push, 1)  // Ensure that struct members are packed tightly

// Define the WAV file header structure
struct WAVHeader {
    char riff_header[4];
    uint32_t wav_size;
    char wave_header[4];
    char fmt_header[4];
    uint32_t fmt_chunk_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

#pragma pack(pop)  // Restore default struct packing

std::vector<float> readWAVFile(const char* filename) {
    // Open the WAV file for binary reading
    std::ifstream wav_file(filename, std::ios::binary);
    
    if (!wav_file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return std::vector<float>();
    }

    // Read the WAV header
    WAVHeader wav_header;
    wav_file.read(reinterpret_cast<char*>(&wav_header), sizeof(wav_header));

    // Check if it's a valid WAV file
    if (strncmp(wav_header.riff_header, "RIFF", 4) != 0 ||
        strncmp(wav_header.wave_header, "WAVE", 4) != 0 ||
        strncmp(wav_header.fmt_header, "fmt ", 4) != 0) {
        std::cerr << "Not a valid WAV file: " << filename << std::endl;
        return std::vector<float>();
    }

    // Determine the audio format
    std::string audio_format_str;
    switch (wav_header.audio_format) {
        case 1:
            audio_format_str = "PCM";
            break;
        case 3:
            audio_format_str = "IEEE Float";
            break;
        default:
            audio_format_str = "Unknown";
            break;
    }

    // Print information from the header
    std::cout << "Audio Format: " << audio_format_str << std::endl;
    std::cout << "Num Channels: " << wav_header.num_channels << std::endl;
    std::cout << "Sample Rate: " << wav_header.sample_rate << std::endl;
    std::cout << "Bits Per Sample: " << wav_header.bits_per_sample << std::endl;

    // Skip any extra bytes in fmt chunk if fmt_chunk_size > 16
    if (wav_header.fmt_chunk_size > 16) {
        wav_file.seekg(wav_header.fmt_chunk_size - 16, std::ios::cur);
    }

    // Find the "data" chunk
    char chunk_id[4];
    uint32_t chunk_size;
    bool found_data = false;
    
    while (wav_file.read(reinterpret_cast<char*>(chunk_id), 4)) {
        wav_file.read(reinterpret_cast<char*>(&chunk_size), 4);
        
        if (strncmp(chunk_id, "data", 4) == 0) {
            found_data = true;
            break;
        } else {
            // Skip this chunk
            wav_file.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!found_data) {
        std::cerr << "Data chunk not found in WAV file" << std::endl;
        wav_file.close();
        return std::vector<float>();
    }

    // Calculate the number of samples
    uint32_t bytes_per_sample = wav_header.bits_per_sample / 8;
    uint32_t num_samples_total = chunk_size / bytes_per_sample;
    uint32_t num_samples_per_channel = num_samples_total / wav_header.num_channels;

    std::vector<float> float_samples;

    if (wav_header.audio_format == 1) { // PCM
        if (wav_header.bits_per_sample == 16) {
            std::vector<int16_t> pcm16_samples(num_samples_total);
            wav_file.read(reinterpret_cast<char*>(pcm16_samples.data()), chunk_size);

            // Convert to float and handle mono/stereo
            if (wav_header.num_channels == 1) {
                // Mono: direct conversion
                float_samples.resize(num_samples_per_channel);
                for (uint32_t i = 0; i < num_samples_per_channel; i++) {
                    float_samples[i] = static_cast<float>(pcm16_samples[i]) / 32768.0f;
                }
            } else {
                // Stereo or multi-channel: convert to mono by averaging
                float_samples.resize(num_samples_per_channel);
                for (uint32_t i = 0; i < num_samples_per_channel; i++) {
                    float sum = 0.0f;
                    for (uint16_t ch = 0; ch < wav_header.num_channels; ch++) {
                        sum += static_cast<float>(pcm16_samples[i * wav_header.num_channels + ch]);
                    }
                    float_samples[i] = (sum / wav_header.num_channels) / 32768.0f;
                }
            }
        } else {
            std::cerr << "Unsupported bits per sample: " << wav_header.bits_per_sample << std::endl;
            wav_file.close();
            return std::vector<float>();
        }
    } else if (wav_header.audio_format == 3) { // IEEE Float
        std::vector<float> float_samples_raw(num_samples_total);
        wav_file.read(reinterpret_cast<char*>(float_samples_raw.data()), chunk_size);

        // Handle mono/stereo
        if (wav_header.num_channels == 1) {
            float_samples = float_samples_raw;
        } else {
            // Convert to mono by averaging
            float_samples.resize(num_samples_per_channel);
            for (uint32_t i = 0; i < num_samples_per_channel; i++) {
                float sum = 0.0f;
                for (uint16_t ch = 0; ch < wav_header.num_channels; ch++) {
                    sum += float_samples_raw[i * wav_header.num_channels + ch];
                }
                float_samples[i] = sum / wav_header.num_channels;
            }
        }
    } else {
        std::cerr << "Unsupported audio format: " << wav_header.audio_format << std::endl;
        wav_file.close();
        return std::vector<float>();
    }

    // Resample to 16kHz if needed (simple linear interpolation)
    if (wav_header.sample_rate != 16000) {
        std::vector<float> resampled;
        float ratio = static_cast<float>(wav_header.sample_rate) / 16000.0f;
        uint32_t new_size = static_cast<uint32_t>(float_samples.size() / ratio);
        resampled.resize(new_size);
        
        for (uint32_t i = 0; i < new_size; i++) {
            float src_index = i * ratio;
            uint32_t src_idx = static_cast<uint32_t>(src_index);
            float frac = src_index - src_idx;
            
            if (src_idx + 1 < float_samples.size()) {
                resampled[i] = float_samples[src_idx] * (1.0f - frac) + float_samples[src_idx + 1] * frac;
            } else {
                resampled[i] = float_samples[src_idx];
            }
        }
        float_samples = resampled;
        std::cout << "Resampled from " << wav_header.sample_rate << " Hz to 16000 Hz" << std::endl;
    }

    // Close the file
    wav_file.close();

    std::cout << "Read " << float_samples.size() << " samples" << std::endl;
    return float_samples;
}