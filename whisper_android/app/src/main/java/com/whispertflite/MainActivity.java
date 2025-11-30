package com.whispertflite;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.whispertflite.asr.IRecorderListener;
import com.whispertflite.asr.IWhisperListener;
import com.whispertflite.asr.Recorder;
import com.whispertflite.asr.Whisper;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.CountDownLatch;

public class MainActivity extends AppCompatActivity {
    private final String TAG = "MainActivity";

    private Whisper mWhisper = null;
    private Recorder mRecorder = null;
    
    // For processing files sequentially
    private Queue<File> audioFileQueue = new LinkedList<>();
    private List<TranscriptionResult> transcriptionResults = new ArrayList<>();
    private CountDownLatch currentFileLatch = null;
    private String currentTranscription = null;
    private boolean isProcessingFiles = false;

    // Transcription result class
    private static class TranscriptionResult {
        String filename;
        String language;
        String transcription;
        long timeMs;

        TranscriptionResult(String filename, String language, String transcription, long timeMs) {
            this.filename = filename;
            this.language = language;
            this.transcription = transcription;
            this.timeMs = timeMs;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // No UI - no setContentView call

        // Call the method to copy specific file types from assets to data folder
        String[] extensionsToCopy = {"pcm", "bin", "wav", "tflite"};
        copyAssetsWithExtensionsToDataFolder(this, extensionsToCopy);

        // Use multilingual model
        String modelPath = getFilePath("whisper-tiny.tflite");
        String vocabPath = getFilePath("filters_vocab_multilingual.bin");
        boolean useMultilingual = true;

        mWhisper = new Whisper(this);
        mWhisper.loadModel(modelPath, vocabPath, useMultilingual);
        mWhisper.setListener(new IWhisperListener() {
            @Override
            public void onUpdateReceived(String message) {
                Log.d(TAG, "Update is received, Message: " + message);

                if (message.equals(Whisper.MSG_PROCESSING)) {
                    // Processing started
                } else if (message.equals(Whisper.MSG_PROCESSING_DONE)) {
                    // Processing done, signal the latch
                    if (currentFileLatch != null) {
                        currentFileLatch.countDown();
                    }
                } else if (message.equals(Whisper.MSG_FILE_NOT_FOUND)) {
                    Log.d(TAG, "File not found error...!");
                    currentTranscription = "";
                    if (currentFileLatch != null) {
                        currentFileLatch.countDown();
                    }
                }
            }

            @Override
            public void onResultReceived(String result) {
                Log.d(TAG, "Result: " + result);
                currentTranscription = result;
            }
        });

        mRecorder = new Recorder(this);
        mRecorder.setListener(new IRecorderListener() {
            @Override
            public void onUpdateReceived(String message) {
                Log.d(TAG, "Update is received, Message: " + message);
            }

            @Override
            public void onDataReceived(float[] samples) {
                //mWhisper.writeBuffer(samples);
            }
        });

        // Check permissions and start processing
        checkPermissions();
    }

    private void checkPermissions() {
        List<String> permissionsNeeded = new ArrayList<>();
        
        // Check RECORD_AUDIO permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) 
                != PackageManager.PERMISSION_GRANTED) {
            permissionsNeeded.add(Manifest.permission.RECORD_AUDIO);
        }
        
        // Check storage permission based on Android version
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ uses READ_MEDIA_AUDIO
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_AUDIO) 
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsNeeded.add(Manifest.permission.READ_MEDIA_AUDIO);
            }
        } else {
            // Older versions use READ_EXTERNAL_STORAGE
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) 
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsNeeded.add(Manifest.permission.READ_EXTERNAL_STORAGE);
            }
        }
        
        if (!permissionsNeeded.isEmpty()) {
            requestPermissions(permissionsNeeded.toArray(new String[0]), 0);
        } else {
            // All permissions granted, start processing
            startProcessingAudioFiles();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        boolean allGranted = true;
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }
        
        if (allGranted) {
            Log.d(TAG, "All permissions granted");
            startProcessingAudioFiles();
        } else {
            Log.e(TAG, "Permissions not granted");
        }
    }

    private void startProcessingAudioFiles() {
        new Thread(() -> {
            try {
                // Wait a bit for model to be fully loaded
                Thread.sleep(1000);
                
                // Scan and process audio files
                scanAndProcessAudioFiles();
            } catch (InterruptedException e) {
                Log.e(TAG, "Error in processing thread", e);
            }
        }).start();
    }

    private void scanAndProcessAudioFiles() {
        Log.d(TAG, "Starting to scan audio files...");
        
        // Get base audio directory (internal storage/audio)
        File baseAudioDir = new File(Environment.getExternalStorageDirectory(), "audio");
        if (!baseAudioDir.exists()) {
            Log.e(TAG, "Audio directory not found: " + baseAudioDir.getAbsolutePath());
            return;
        }

        // Define language directories
        // String[] languageDirs = {"english", "french"};
        String[] languageDirs = {"arabic","farsi"};
        
        // Collect all audio files
        for (String langDir : languageDirs) {
            File langPath = new File(baseAudioDir, langDir);
            if (langPath.exists() && langPath.isDirectory()) {
                Log.d(TAG, "Scanning directory: " + langPath.getAbsolutePath());
                List<File> audioFiles = findAudioFiles(langPath);
                for (File audioFile : audioFiles) {
                    audioFileQueue.add(audioFile);
                    Log.d(TAG, "Added file to queue: " + audioFile.getName() + " (language: " + langDir + ")");
                }
            } else {
                Log.w(TAG, "Language directory not found: " + langPath.getAbsolutePath());
            }
        }

        if (audioFileQueue.isEmpty()) {
            Log.w(TAG, "No audio files found to process");
            return;
        }

        Log.d(TAG, "Found " + audioFileQueue.size() + " audio files to process");
        isProcessingFiles = true;
        
        // Process files sequentially
        while (!audioFileQueue.isEmpty() && isProcessingFiles) {
            File audioFile = audioFileQueue.poll();
            String language = getLanguageFromPath(audioFile.getAbsolutePath());
            processAudioFile(audioFile, language);
        }
        
        // Generate JSON output
        generateJSONOutput();
    }

    private List<File> findAudioFiles(File directory) {
        List<File> audioFiles = new ArrayList<>();
        File[] files = directory.listFiles();
        
        if (files != null) {
            for (File file : files) {
                if (file.isFile()) {
                    String fileName = file.getName().toLowerCase();
                    if (fileName.endsWith(".wav") || fileName.endsWith(".mp3") || 
                        fileName.endsWith(".m4a") || fileName.endsWith(".flac") ||
                        fileName.endsWith(".ogg")) {
                        audioFiles.add(file);
                    }
                } else if (file.isDirectory()) {
                    // Recursively search subdirectories
                    audioFiles.addAll(findAudioFiles(file));
                }
            }
        }
        
        return audioFiles;
    }

    private String getLanguageFromPath(String filePath) {
        // Normalize path separators for cross-platform compatibility
        String normalizedPath = filePath.replace('\\', '/').toLowerCase();
        
        if (normalizedPath.contains("/english/")) {
            return "english";
        } else if (normalizedPath.contains("/french/")) {
            return "french";
        } else if (normalizedPath.contains("/arabic/")) {
            return "arabic";
        }
        return "unknown";
    }

    private void processAudioFile(File audioFile, String language) {
        Log.d(TAG, "Processing file: " + audioFile.getName() + " (language: " + language + ")");
        
        currentTranscription = null;
        currentFileLatch = new CountDownLatch(1);
        
        // Record start time
        long startTime = System.currentTimeMillis();
        
        // Start transcription
        mWhisper.setFilePath(audioFile.getAbsolutePath());
        mWhisper.setAction(Whisper.ACTION_TRANSCRIBE);
        mWhisper.start();
        
        try {
            // Wait for transcription to complete (with timeout)
            boolean completed = currentFileLatch.await(300, java.util.concurrent.TimeUnit.SECONDS);
            
            // Calculate elapsed time
            long elapsedTime = System.currentTimeMillis() - startTime;
            
            if (completed && currentTranscription != null) {
                // Clean the transcription (remove extra tokens)
                String cleanedTranscription = cleanTranscription(currentTranscription);
                transcriptionResults.add(new TranscriptionResult(
                    audioFile.getName(),
                    language,
                    cleanedTranscription,
                    elapsedTime
                ));
                Log.d(TAG, "Transcription completed for: " + audioFile.getName() + " in " + elapsedTime + "ms");
            } else {
                Log.e(TAG, "Transcription timeout or failed for: " + audioFile.getName());
                transcriptionResults.add(new TranscriptionResult(
                    audioFile.getName(),
                    language,
                    "",
                    elapsedTime
                ));
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while waiting for transcription", e);
            long elapsedTime = System.currentTimeMillis() - startTime;
            transcriptionResults.add(new TranscriptionResult(
                audioFile.getName(),
                language,
                "",
                elapsedTime
            ));
        }
        
        // Wait a bit before processing next file
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted", e);
        }
    }

    private void generateJSONOutput() {
        try {
            JSONArray jsonArray = new JSONArray();
            
            for (TranscriptionResult result : transcriptionResults) {
                JSONObject jsonObject = new JSONObject();
                jsonObject.put("filename", result.filename);
                jsonObject.put("language", result.language);
                jsonObject.put("transcription", result.transcription);
                jsonObject.put("timeMs", result.timeMs);
                jsonArray.put(jsonObject);
            }
            
            // Save JSON to file in app's internal storage
            File outputFile = new File(getFilesDir(), "transcriptions.json");
            FileWriter fileWriter = new FileWriter(outputFile);
            fileWriter.write(jsonArray.toString(2)); // Pretty print with 2-space indent
            fileWriter.close();
            
            Log.d(TAG, "JSON output saved to: " + outputFile.getAbsolutePath());
            Log.d(TAG, "Total transcriptions: " + transcriptionResults.size());
            
            // Close the app after all transcriptions are complete
            Log.d(TAG, "All transcriptions completed. Closing app...");
            finish();
            System.exit(0);
            
        } catch (Exception e) {
            Log.e(TAG, "Error generating JSON output", e);
            finish();
            System.exit(1);
        }
    }
    
    private String cleanTranscription(String transcription) {
        if (transcription == null || transcription.isEmpty()) {
            return transcription;
        }
        
        // Remove extra tokens and special tokens
        String cleaned = transcription
            .replaceAll("\\[_extra_token_\\d+\\]", "")  // Remove [_extra_token_XXXXX]
            .replaceAll("\\[_TT_\\d+\\]", "")           // Remove [_TT_XXXXX] timestamp tokens
            .replaceAll("\\[_EOT_\\]", "")               // Remove [_EOT_]
            .replaceAll("\\[_SOT_\\]", "")               // Remove [_SOT_]
            .replaceAll("\\[_PREV_\\]", "")              // Remove [_PREV_]
            .replaceAll("\\[_NOT_\\]", "")               // Remove [_NOT_]
            .replaceAll("\\[_BEG_\\]", "")               // Remove [_BEG_]
            .trim();                                     // Trim whitespace
        
        return cleaned;
    }

    // Recording calls
    private void startRecording() {
        checkPermissions();
        // Recording functionality can be called programmatically if needed
    }

    private void stopRecording() {
        if (mRecorder != null) {
            mRecorder.stop();
        }
    }

    // Copy assets to data folder
    private static void copyAssetsWithExtensionsToDataFolder(Context context, String[] extensions) {
        AssetManager assetManager = context.getAssets();
        try {
            // Specify the destination directory in the app's data folder
            String destFolder = context.getFilesDir().getAbsolutePath();

            for (String extension : extensions) {
                // List all files in the assets folder with the specified extension
                String[] assetFiles = assetManager.list("");
                for (String assetFileName : assetFiles) {
                    if (assetFileName.endsWith("." + extension)) {
                        File outFile = new File(destFolder, assetFileName);
                        if (outFile.exists())
                            continue;

                        InputStream inputStream = assetManager.open(assetFileName);
                        OutputStream outputStream = new FileOutputStream(outFile);

                        // Copy the file from assets to the data folder
                        byte[] buffer = new byte[1024];
                        int read;
                        while ((read = inputStream.read(buffer)) != -1) {
                            outputStream.write(buffer, 0, read);
                        }

                        inputStream.close();
                        outputStream.flush();
                        outputStream.close();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Returns file path from data folder
    private String getFilePath(String assetName) {
        File outfile = new File(getFilesDir(), assetName);
        if (!outfile.exists()) {
            Log.d(TAG, "File not found - " + outfile.getAbsolutePath());
        }

        Log.d(TAG, "Returned asset path: " + outfile.getAbsolutePath());
        return outfile.getAbsolutePath();
    }
}
