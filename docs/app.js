let model;
let audioContext;
let microphone;
let processor;
let isRecording = false;
let isLoaded = false;
let isFileModelsLoaded = false;
let startStopBtn;
let loadBtn;
let loadFileModelsBtn;
let animationId;
let sensitivitySlider;
let sensitivityValue;
let detectionList;
let dogLogBody;
let totalRecordingsDisplay;
let sensitivity = 0.3;
let lastDetectionTime = 0;
let lastLoggedDetectionTime = 0;
let mediaStream = null;
let saveAsHtmlBtn;
let saveAsCsvBtn;
let shareBtn;
let shareAppBtn;
let micHelpText;
let hasShared = false;
let lastShareUrl = '';
let audioTickCount = 0;
let wakeAudioSource = null;
let wakeLock = null;
let totalRecordings = 0;
let fileInput;
let fileInfo;
let processFileBtn;
let fileProgress;
let fileProgressFill;
let progressText;
let dropZone;
let cancelProcessBtn;
let selectedFile = null;
let isProcessingFile = false;
let fileCurrentTime = 0; // Current time position in seconds during file processing
let currentPlayingButton = null; // Track currently playing audio button
const WINDOW_SIZE = 15360; // 0.96s at 16kHz
const HOP_SIZE = 7680; // 0.48s at 16kHz
const SAVE_BUFFER_SIZE = 32000; // 2s at 16kHz for full bark capture
let windowBuffer = new Float32Array(WINDOW_SIZE);
let saveBuffer = new Float32Array(SAVE_BUFFER_SIZE);
let saveIndex = 0;
let bufferFilled = false;
let windowPosition = 0; // Track position in the window
let predictionBuffer = []; // Buffer to hold predictions over 0.96s period
let predictionCount = 0; // Counter for predictions in current period

function updateProgressBar(percentage) {
    const progressFill = loadBtn.querySelector('.progress-fill');
    if (progressFill) {
        progressFill.style.width = percentage + '%';
    }
}

async function init() {
    startStopBtn = document.getElementById('startStopBtn');
    loadBtn = document.getElementById('loadBtn');
    loadFileModelsBtn = document.getElementById('loadFileModelsBtn');
    sensitivitySlider = document.getElementById('sensitivity');
    sensitivityValue = document.getElementById('sensitivityValue');
    detectionList = document.getElementById('detectionList');
    dogLogBody = document.getElementById('dogLogBody');
    totalRecordingsDisplay = document.getElementById('totalRecordings');
    saveAsHtmlBtn = document.getElementById('saveAsHtmlBtn');
    saveAsCsvBtn = document.getElementById('saveAsCsvBtn');
    shareBtn = document.getElementById('shareBtn');
    shareAppBtn = document.getElementById('shareAppBtn');
    micHelpText = document.getElementById('micHelpText');
    fileInput = document.getElementById('fileInput');
    fileInfo = document.getElementById('fileInfo');
    processFileBtn = document.getElementById('processFileBtn');
    fileProgress = document.getElementById('fileProgress');
    fileProgressFill = document.getElementById('fileProgressFill');
    progressText = document.getElementById('progressText');
    dropZone = document.getElementById('dropZone');
    cancelProcessBtn = document.getElementById('cancelProcessBtn');

    loadBtn.addEventListener('click', loadModels);
    loadFileModelsBtn.addEventListener('click', loadFileModels);
    startStopBtn.addEventListener('click', toggleRecording);
    sensitivitySlider.addEventListener('input', updateSensitivity);
    saveAsHtmlBtn.addEventListener('click', saveAsHtml);
    saveAsCsvBtn.addEventListener('click', saveAsCsv);
    shareBtn.addEventListener('click', handleShareClick);
    shareAppBtn.addEventListener('click', shareApp);
    fileInput.addEventListener('change', handleFileSelect);
    processFileBtn.addEventListener('click', processFile);
    cancelProcessBtn.addEventListener('click', cancelProcessing);

    // Drag and drop functionality
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    // Add placeholder entries to dog log immediately
    addPlaceholderEntries();

    // Initialize detection list with placeholder values
    updateDetectionList([]);

    // Initialize total recordings display
    updateTotalRecordings();

    // Initialize modal event listeners
    initModalListeners();

    // If models are already loaded (from a previous session or shared loading), update button states
    if (isLoaded) {
        loadBtn.querySelector('.button-text').textContent = 'Loaded ✓';
        loadBtn.disabled = true;
        startStopBtn.disabled = false;
        isFileModelsLoaded = true;
    }
    if (isFileModelsLoaded) {
        loadFileModelsBtn.querySelector('.button-text').textContent = 'Models Loaded ✓';
        loadFileModelsBtn.disabled = true;
        if (selectedFile) {
            processFileBtn.disabled = false;
            processFileBtn.textContent = 'Start Processing';
        }
    }
}

async function loadModels() {
    try {
        loadBtn.disabled = true;
        loadBtn.classList.add('loading');
        loadBtn.querySelector('.button-text').textContent = 'Initializing...';
        micHelpText.style.display = 'none';
        updateProgressBar(10);

        log('Initializing...');

        // Set WASM backend explicitly for consistent performance across devices
        await tf.setBackend('wasm');
        await tf.ready();
        log('WASM backend ready');
        loadBtn.querySelector('.button-text').textContent = 'Setting up TensorFlow...';
        micHelpText.style.display = 'none';
        updateProgressBar(25);

        loadBtn.querySelector('.button-text').textContent = 'Loading YAMNet model...';
        micHelpText.style.display = 'none';
        updateProgressBar(40);
        model = await yamnet.load('./model/', {
            requestInit: {
                cache: 'no-cache'
            }
        });
        log('YAMNet model loaded');
        updateProgressBar(60);

        // Request microphone access with simplified constraints
        loadBtn.querySelector('.button-text').textContent = 'Requesting microphone access...';
        micHelpText.style.display = 'block';
        updateProgressBar(70);
        const constraints = {
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: true,  // Enable AGC to boost low mobile audio
                channelCount: 1,
                sampleRate: { ideal: 16000 }
            }
        };

        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        log('Microphone ready');
        loadBtn.querySelector('.button-text').textContent = 'Setting up audio processing...';
        micHelpText.style.display = 'none';
        updateProgressBar(80);

        // Create AudioContext
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        window.audioContextInstance = audioContext;

        // Create microphone source
        microphone = audioContext.createMediaStreamSource(mediaStream);


        // Create AudioWorklet processor
        await audioContext.audioWorklet.addModule('audio-processor.js');
        processor = new AudioWorkletNode(audioContext, 'audio-processor');
        microphone.connect(processor);
        processor.connect(audioContext.destination);

        // Setup audio processing handler
        let audioBuffer = [];
        processor.port.onmessage = (event) => {
            if (event.data.type === 'audioData' && isRecording) {
                audioBuffer.push(...event.data.data);

                // Process in small chunks to maintain real-time processing
                const sampleRate = audioContext.sampleRate;
                const chunkSize = Math.floor(sampleRate * 0.1); // 100ms chunks

                while (audioBuffer.length >= chunkSize) {
                    const chunk = audioBuffer.splice(0, chunkSize);
                    audioTickCount++;

                    // Resample chunk to 16kHz first
                    const resampledChunk = resampleAudio(chunk, sampleRate, 16000);

                    // Add resampled data to save buffer for full bark capture
                    for (let i = 0; i < resampledChunk.length; i++) {
                        saveBuffer[saveIndex] = resampledChunk[i];
                        saveIndex = (saveIndex + 1) % SAVE_BUFFER_SIZE;
                        if (saveIndex === 0) bufferFilled = true;
                    }

                    // Add resampled data to window buffer
                    processAudioChunk(resampledChunk, 16000);
                }
            }
        };

        loadBtn.querySelector('.button-text').textContent = 'Warming up model...';
        micHelpText.style.display = 'none';
        updateProgressBar(90);
        // Warm up YAMNet inference engine with a test prediction
        const testBuffer = new Float32Array(16000).fill(0);
        const warmupPredictions = await model.predict(testBuffer);
        warmupPredictions.dispose();

        // Everything loaded successfully
        isLoaded = true;
        isFileModelsLoaded = true; // Model is now available for file processing too
        loadBtn.classList.remove('loading');
        loadBtn.querySelector('.button-text').textContent = 'Loaded ✓';
        micHelpText.style.display = 'none';
        loadBtn.disabled = true;
        startStopBtn.disabled = false;

        // Enable file processing if a file is selected
        if (selectedFile) {
            processFileBtn.disabled = false;
            processFileBtn.textContent = 'Start Processing';
        }

        updateProgressBar(100);

        // Keep progress bar visible briefly after completion
        setTimeout(() => {
            updateProgressBar(0);
        }, 2000);

        log('Ready to record');
        showToast('Models loaded! Ready to record.');

    } catch (error) {
        const errorDetails = `Name: ${error.name || 'Unknown'}, Message: ${error.message || error.toString() || 'No message'}, Stack: ${error.stack ? error.stack.split('\n')[0] : 'No stack'}`;
        log(`ERROR: ${error.message}`);

        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            showToast('Microphone access denied. Please allow access and try again.');
        } else {
            showToast(`Error: ${error.message || error.toString()}`);
        }

        loadBtn.disabled = false;
        loadBtn.classList.remove('loading');
        loadBtn.querySelector('.button-text').textContent = 'Load Models & Setup Mic';
        micHelpText.style.display = 'none';
        updateProgressBar(0);

        // Reset progress bar on error
        setTimeout(() => {
            updateProgressBar(0);
        }, 100);
    }
}

async function loadFileModels() {
    try {
        loadFileModelsBtn.disabled = true;
        loadFileModelsBtn.classList.add('loading');
        loadFileModelsBtn.querySelector('.button-text').textContent = 'Initializing...';
        updateProgressBar(10);

        log('Initializing file models...');

        // Set WASM backend explicitly for consistent performance across devices
        await tf.setBackend('wasm');
        await tf.ready();
        log('WASM backend ready for files');
        loadFileModelsBtn.querySelector('.button-text').textContent = 'Setting up TensorFlow...';
        updateProgressBar(25);

        // Check if model is already loaded from main mic setup
        if (!model) {
            loadFileModelsBtn.querySelector('.button-text').textContent = 'Loading YAMNet model...';
            updateProgressBar(40);
            model = await yamnet.load('./model/', {
                requestInit: {
                    cache: 'no-cache'
                }
            });
            log('YAMNet model loaded for files');
            updateProgressBar(60);
        } else {
            log('YAMNet model already loaded, skipping...');
            updateProgressBar(60);
        }

        loadFileModelsBtn.querySelector('.button-text').textContent = 'Warming up model...';
        updateProgressBar(80);
        // Warm up YAMNet inference engine with a test prediction
        const testBuffer = new Float32Array(16000).fill(0);
        const warmupPredictions = await model.predict(testBuffer);
        warmupPredictions.dispose();

        // Everything loaded successfully
        isFileModelsLoaded = true;
        loadFileModelsBtn.classList.remove('loading');
        loadFileModelsBtn.querySelector('.button-text').textContent = 'Models Loaded ✓';
        loadFileModelsBtn.disabled = true;

        // Enable file processing if a file is selected
        if (selectedFile) {
            processFileBtn.disabled = false;
            processFileBtn.textContent = 'Start Processing';
        }

        updateProgressBar(100);

        // Keep progress bar visible briefly after completion
        setTimeout(() => {
            updateProgressBar(0);
        }, 2000);

        log('File models ready');
        showToast('File models loaded! Ready to process files.');

    } catch (error) {
        const errorDetails = `Name: ${error.name || 'Unknown'}, Message: ${error.message || error.toString() || 'No message'}, Stack: ${error.stack ? error.stack.split('\n')[0] : 'No stack'}`;
        log(`ERROR loading file models: ${error.message}`);

        showToast(`Error loading models: ${error.message || error.toString()}`);

        loadFileModelsBtn.disabled = false;
        loadFileModelsBtn.classList.remove('loading');
        loadFileModelsBtn.querySelector('.button-text').textContent = 'Load Models';
        updateProgressBar(0);

        // Reset progress bar on error
        setTimeout(() => {
            updateProgressBar(0);
        }, 100);
    }
}

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    if (!isLoaded) {
        showToast('Please load models first');
        return;
    }

    try {
        log('Recording started');

        // Show UI sections
        document.getElementById('detection').style.display = 'block';
        document.getElementById('detectionPlaceholder').style.display = 'none';
        document.getElementById('dogLog').style.display = 'block';

        // Enable recording
        isRecording = true;
        audioTickCount = 0;
        // Reset buffers
        windowBuffer = new Float32Array(WINDOW_SIZE);
        saveBuffer = new Float32Array(SAVE_BUFFER_SIZE);
        saveIndex = 0;
        bufferFilled = false;
        windowPosition = 0;
        predictionBuffer = [];
        predictionCount = 0;

        // Disable file processing during live recording
        processFileBtn.disabled = true;
        if (selectedFile) {
            processFileBtn.textContent = 'Recording in progress';
        }

        // Request wake lock to prevent device sleep, fallback to silent audio
        if ('wakeLock' in navigator) {
            try {
                wakeLock = await navigator.wakeLock.request('screen');
            } catch (error) {
                startWakeAudio();
            }
        } else {
            startWakeAudio();
        }

        startStopBtn.textContent = 'Stop Recording';
        startStopBtn.classList.add('stop-btn');

        // Already logged above

    } catch (error) {
        log(`Recording error: ${error.message}`);
        showToast(`Error: ${error.message}`);
    }
}

function stopRecording() {
    log('Recording stopped');

    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }

    isRecording = false;
    startStopBtn.classList.remove('stop-btn');
    startStopBtn.textContent = 'Start Recording';

    // Keep sections visible
    document.getElementById('detectionPlaceholder').style.display = 'block';

    // Release wake lock or stop silent audio loop
    if (wakeLock) {
        try {
            wakeLock.release();
            wakeLock = null;
        } catch (error) {
            // Ignore release errors
        }
    } else {
        stopWakeAudio();
    }

    // Re-enable file processing if models are loaded and file is selected
    if (isLoaded && selectedFile) {
        processFileBtn.disabled = false;
        processFileBtn.textContent = 'Start Processing';
    }

    // Already logged above
}

function processAudioChunk(audioData, sampleRate) {
    try {
        // Process audio data and fill both buffers simultaneously
        for (let i = 0; i < audioData.length; i++) {
            // Add to save buffer first (for bark capture)
            saveBuffer[saveIndex] = audioData[i];
            saveIndex = (saveIndex + 1) % SAVE_BUFFER_SIZE;
            if (saveIndex === 0) bufferFilled = true;

            // Add to window buffer for YAMNet processing
            if (windowPosition < WINDOW_SIZE) {
                windowBuffer[windowPosition] = audioData[i];
                windowPosition++;
            } else {
                // Window is full, shift by hop size and add new data
                windowBuffer.copyWithin(0, HOP_SIZE);
                windowPosition = WINDOW_SIZE - HOP_SIZE;
                windowBuffer[windowPosition] = audioData[i];
                windowPosition++;

                // Only process every full window (every 0.96s), not every hop
                // This ensures we process once per complete window, not every 0.48s
                processYamnetWindow(windowBuffer.slice());
                predictionCount++;
            }
        }

    } catch (error) {
        log(`ERROR Processing chunk: ${error.message}`);
    }
}

function resampleAudio(audioData, fromRate, toRate) {
    const ratio = toRate / fromRate;
    const newLength = Math.floor(audioData.length * ratio);
    const resampled = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
        const index = i / ratio;
        const low = Math.floor(index);
        const high = Math.min(low + 1, audioData.length - 1);
        const weight = index - low;
        resampled[i] = audioData[low] * (1 - weight) + audioData[high] * weight;
    }

    return resampled;
}

async function processYamnetWindow(yamnetInput) {
    try {
        // Amplitude normalization (remove DC offset and normalize)
        const mean = yamnetInput.reduce((sum, val) => sum + val, 0) / yamnetInput.length;
        const dcRemoved = yamnetInput.map(val => val - mean);
        const maxAbs = Math.max(...dcRemoved.map(Math.abs));
        const normalized = maxAbs > 0 ? dcRemoved.map(val => val / maxAbs) : dcRemoved;

        // Run YAMNet inference
        const yamnetTensor = tf.tensor(normalized, [WINDOW_SIZE], 'float32');
        const predictions = await model.predict(yamnetTensor);
        const scores = await predictions.data();

        yamnetTensor.dispose();

        // Store predictions in buffer
        predictionBuffer.push(scores);

        // Check if we have 2 predictions (0.96s worth: 0.48s hop * 2 = 0.96s)
        if (predictionBuffer.length >= 2) {
            // Combine predictions by averaging
            const combinedScores = new Float32Array(scores.length);
            for (let i = 0; i < scores.length; i++) {
                combinedScores[i] = predictionBuffer.reduce((sum, pred) => sum + pred[i], 0) / predictionBuffer.length;
            }

            // Get top 5 classes for display from combined scores
            const topClasses = [];
            for (let i = 0; i < Math.min(combinedScores.length, 521); i++) {
                topClasses.push({
                    index: i,
                    score: combinedScores[i],
                    name: model.classNames[i] || `Class ${i}`
                });
            }
            topClasses.sort((a, b) => b.score - a.score);

            // Update UI with top detections
            updateDetectionList(topClasses);

            // Log all YAMNet predictions for debugging
            const allPredictions = Array.from(combinedScores)
                .map((score, index) => ({ score, index }))
                .filter(item => item.score > 0.01)
                .sort((a, b) => b.score - a.score)
                .slice(0, 10) // Top 10 predictions
                .map(item => `${model.classNames[item.index]} (${(item.score * 100).toFixed(1)}%)`)
                .join(', ');
            log(`YAMNet: ${allPredictions}`);

            // Check for dog sounds using combined scores (classes: Animal=67, Domestic animals=68, Dog=69, Bark=70, Yip=71, Howl=72, Bow-wow=73, Growling=74, Whimper=75, Canidae=117)
            const dogClasses = [67, 68, 69, 70, 71, 72, 73, 74, 75, 117]; // Animal, Domestic animals, Dog, Bark, Yip, Howl, Bow-wow, Growling, Whimper, Canidae
            const detectedDogs = dogClasses
                .filter(classIdx => combinedScores[classIdx] > sensitivity)
                .map(classIdx => ({
                    name: model.classNames[classIdx],
                    score: combinedScores[classIdx]
                }));

            // Check for specific dog sounds (excluding general Animal and Domestic animals classes)
            const specificDogClasses = [69, 70, 71, 72, 73, 74, 75, 117]; // Dog, Bark, Yip, Howl, Bow-wow, Growling, Whimper, Canidae
            const hasSpecificDogSound = specificDogClasses.some(classIdx => combinedScores[classIdx] > 0.01); // >1%

            // Check for general animal classes above sensitivity threshold
            const hasGeneralAnimalClass = (combinedScores[67] > sensitivity) || (combinedScores[68] > sensitivity); // Animal or Domestic animals

            // Check for dog detection: must have specific dog sound >1% AND (specific dog class above threshold OR general animal class above threshold)
            const shouldDetectDog = hasSpecificDogSound && (detectedDogs.length > 0 || hasGeneralAnimalClass);

            if (shouldDetectDog) {
                const currentTime = isProcessingFile ? fileCurrentTime : (Date.now() / 1000); // Use file time or real time
                const timeSinceLastLog = currentTime - lastLoggedDetectionTime;

                // Get all dog-related classes above 1% threshold for logging
                const allDogClasses = [67, 68, 69, 70, 71, 72, 73, 74, 75, 117]; // Animal, Domestic animals, Dog, Bark, Yip, Howl, Bow-wow, Growling, Whimper, Canidae
                let relevantClasses = allDogClasses
                    .filter(classIdx => combinedScores[classIdx] > 0.01) // Show classes above 1% threshold
                    .map(classIdx => ({
                        name: model.classNames[classIdx],
                        score: combinedScores[classIdx]
                    }))
                    .sort((a, b) => b.score - a.score); // Sort by confidence descending

                const soundList = relevantClasses.map(item => `${item.name} (${(item.score * 100).toFixed(1)}%)`).join(', ');

                // For file processing, no cooldown - log every detection
                if (isProcessingFile || timeSinceLastLog >= 4.0) {
                    // Log to console and add to dog log
                    log(`Detection: ${soundList}`);
                    if (!isProcessingFile) {
                        showToast('Detection logged!');
                    }

                    // Temporarily update the page title to 'Captured!' (only for live recording)
                    if (!isProcessingFile) {
                        const originalTitle = document.title;
                        document.title = 'Captured!';
                        // Revert title after toast duration (1.5 seconds)
                        setTimeout(() => {
                            document.title = originalTitle;
                        }, 1500);
                    }

                    // Capture full 2-second buffer around detection
                    const capturedAudio = new Float32Array(SAVE_BUFFER_SIZE);
                    if (bufferFilled) {
                        // Buffer has wrapped around - get the most recent 2 seconds
                        const startIdx = saveIndex;
                        for (let i = 0; i < SAVE_BUFFER_SIZE; i++) {
                            capturedAudio[i] = saveBuffer[(startIdx + i) % SAVE_BUFFER_SIZE];
                        }
                    } else {
                        // Buffer hasn't wrapped yet - pad with zeros if needed
                        capturedAudio.set(saveBuffer.slice(0, saveIndex));
                        // Pad the rest with zeros to make it 2 seconds
                        for (let i = saveIndex; i < SAVE_BUFFER_SIZE; i++) {
                            capturedAudio[i] = 0;
                        }
                    }


                    // Create timestamp string
                    let timestamp;
                    if (isProcessingFile) {
                        // Format as MM:SS for file processing
                        const minutes = Math.floor(fileCurrentTime / 60);
                        const seconds = Math.floor(fileCurrentTime % 60);
                        timestamp = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')} (file)`;
                    } else {
                        timestamp = new Date().toLocaleTimeString();
                    }

                    addToDogLog(timestamp, soundList, '', capturedAudio, 16000);
                    lastLoggedDetectionTime = currentTime;
                } else {
                    // Log to console with cooldown message, but don't add to dog log
                    log(`Detection Within ${cooldownTime} Second Cooldown: ${soundList}`);
                }
            }

            // Clear buffer for next 0.96s period
            predictionBuffer = [];
        }

        predictions.dispose();

    } catch (error) {
        log(`ERROR Processing YAMNet: ${error.message}`);
    }
}


function updateSensitivity() {
    sensitivity = parseFloat(sensitivitySlider.value);
    sensitivityValue.textContent = sensitivity.toFixed(1);
}

function updateTotalRecordings() {
    totalRecordingsDisplay.textContent = `Total Recordings = ${totalRecordings}`;
}

function updateDetectionList(topClasses) {
    detectionList.innerHTML = '';

    // If not recording, show placeholder values
    if (!isRecording) {
        const placeholderClasses = [
            { name: 'Dog', score: 0.55 },
            { name: 'Birds', score: 0.10 },
            { name: 'Animal', score: 0.05 },
            { name: 'Typing', score: 0.045 },
            { name: 'Wind', score: 0.03 }
        ];
        placeholderClasses.forEach((classInfo, index) => {
            const detectionItem = document.createElement('div');
            detectionItem.className = 'detection-item placeholder-item';
            detectionItem.innerHTML = `
                <span class="class-name">${index + 1}. ${classInfo.name}</span>
                <span class="confidence">${(classInfo.score * 100).toFixed(1)}%</span>
            `;
            detectionList.appendChild(detectionItem);
        });
    } else {
        // Show actual detection results when recording
        const dogClasses = [67, 68, 69, 70, 71, 72, 73, 74, 75, 117]; // Animal, Domestic animals, Dog, Bark, Yip, Howl, Bow-wow, Growling, Whimper, Canidae
        topClasses.slice(0, 5).forEach((classInfo, index) => {
            const detectionItem = document.createElement('div');
            detectionItem.className = 'detection-item';
            // Check if this is a dog-related class and above tolerance (sensitivity threshold)
            if (dogClasses.includes(classInfo.index) && classInfo.score > sensitivity) {
                detectionItem.classList.add('above-tolerance');
            } else if (dogClasses.includes(classInfo.index) && classInfo.score > 0.01) {
                detectionItem.classList.add('below-tolerance');
            }
            detectionItem.innerHTML = `
                <span class="class-name">${index + 1}. ${classInfo.name}</span>
                <span class="confidence">${(classInfo.score * 100).toFixed(1)}%</span>
            `;
            detectionList.appendChild(detectionItem);
        });
    }

    // If this is the first detection update and we're still in loading state, enable the button and change to stop recording
    if (isRecording && startStopBtn.disabled && startStopBtn.classList.contains('loading')) {
        startStopBtn.disabled = false;
        startStopBtn.classList.remove('loading');
        startStopBtn.textContent = 'Stop Recording';
        startStopBtn.classList.add('stop-btn');
    }
}

function addPlaceholderEntries() {
    for (let i = 0; i < 1; i++) {
        const row = document.createElement('tr');
        row.className = 'placeholder-row';
        row.innerHTML = `
            <td>—</td>
            <td>Waiting for detection...</td>
            <td>—</td>
            <td>—</td>
        `;
        dogLogBody.appendChild(row);
    }
}

// Placeholder entries are now added inside init

function deleteRow(row) {
    row.remove();
    // Update total recordings counter
    totalRecordings--;
    updateTotalRecordings();
}

function addToDogLog(timestamp, soundName, confidence, audioData, sampleRate) {
    // Remove placeholder rows when first real entry is added
    const placeholders = dogLogBody.querySelectorAll('.placeholder-row');
    placeholders.forEach(placeholder => placeholder.remove());

    const row = document.createElement('tr');
    const now = new Date();
    const day = now.getDate();
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = monthNames[now.getMonth()];
    const year = now.getFullYear();
    const hours = now.getHours();
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const seconds = now.getSeconds().toString().padStart(2, '0');
    const ampm = hours >= 12 ? 'pm' : 'am';
    const displayHours = hours % 12 || 12;

    // Format day with ordinal suffix
    const getOrdinalSuffix = (day) => {
        if (day > 3 && day < 21) return 'th';
        switch (day % 10) {
            case 1: return 'st';
            case 2: return 'nd';
            case 3: return 'rd';
            default: return 'th';
        }
    };

    const formattedTimestamp = `${day}${getOrdinalSuffix(day)} ${month} ${year}, ${displayHours}:${minutes}:${seconds}${ampm}`;

    const playButton = document.createElement('button');
    playButton.innerHTML = '▶';
    playButton.className = 'play-btn';
    playButton.addEventListener('click', () => {
        // Reset any currently playing button
        if (currentPlayingButton && currentPlayingButton !== playButton) {
            currentPlayingButton.innerHTML = '▶';
        }
        currentPlayingButton = playButton;
        playAudio(audioData, sampleRate);
    });

    const deleteButton = document.createElement('button');
    deleteButton.innerHTML = '×';
    deleteButton.className = 'delete-btn';
    deleteButton.addEventListener('click', () => {
        deleteRow(row);
    });

    row.innerHTML = `
        <td>${formattedTimestamp}</td>
        <td>${soundName}</td>
    `;
    const playCell = document.createElement('td');
    playCell.appendChild(playButton);
    row.appendChild(playCell);

    const deleteCell = document.createElement('td');
    deleteCell.appendChild(deleteButton);
    row.appendChild(deleteCell);

    // Store audio data on the row for saving
    row.audioData = audioData;
    row.sampleRate = sampleRate;
    row.soundName = soundName;
    row.timestamp = formattedTimestamp;

    dogLogBody.appendChild(row);

    // Update total recordings counter
    totalRecordings++;
    updateTotalRecordings();
}

function log(message) {
    console.log(`[DogBarkDetector] ${message}`);
}

function startWakeAudio() {
    try {
        // Create a silent audio buffer
        const bufferSize = audioContext.sampleRate * 2; // 2 seconds of audio
        const buffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
        const data = buffer.getChannelData(0);

        // Fill with silence (very low amplitude noise to prevent optimization)
        for (let i = 0; i < bufferSize; i++) {
            data[i] = Math.random() * 0.0001; // Very quiet noise
        }

        // Create source and connect to destination
        wakeAudioSource = audioContext.createBufferSource();
        wakeAudioSource.buffer = buffer;
        wakeAudioSource.loop = true;
        wakeAudioSource.connect(audioContext.destination);

        // Start playing
        wakeAudioSource.start();
        log('Wake audio started');
    } catch (error) {
        log(`ERROR: Failed to start wake audio - ${error.message}`);
    }
}

function stopWakeAudio() {
    try {
        if (wakeAudioSource) {
            wakeAudioSource.stop();
            wakeAudioSource.disconnect();
            wakeAudioSource = null;
            log('Wake audio stopped');
        }
    } catch (error) {
        log(`ERROR: Failed to stop wake audio - ${error.message}`);
    }
}



function showToast(message) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.style.display = 'block';

    // Remove any existing fade-out class
    toast.classList.remove('fade-out');

    // Hide after 1.5 seconds
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => {
            toast.style.display = 'none';
        }, 300); // Match animation duration
    }, 1500);
}






async function playAudio(audioData, sampleRate) {
    try {
        // Normalize audio to prevent quiet playback
        const maxAmp = Math.max(...audioData.map(Math.abs));
        if (maxAmp > 0) {
            for (let i = 0; i < audioData.length; i++) {
                audioData[i] = audioData[i] / maxAmp;
            }
        }

        // Convert Float32Array to 16-bit PCM
        const pcmData = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            pcmData[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
        }

        // Create WAV file in memory
        const wavBuffer = createWAVBuffer(pcmData, sampleRate);
        const blob = new Blob([wavBuffer], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(blob);

        // Use HTML5 Audio element for playback
        const audio = new Audio(audioUrl);
        audio.volume = 1.0;

        audio.oncanplay = () => {
            audio.play().then(() => {
                // Change button to pause icon when playing starts
                if (currentPlayingButton) {
                    currentPlayingButton.innerHTML = '⏸';
                }
            }).catch(err => {
                log(`ERROR: Audio playback failed - ${err.message}`);
            });
        };

        audio.onended = () => {
            // Reset button to play icon when audio ends
            if (currentPlayingButton) {
                currentPlayingButton.innerHTML = '▶';
                currentPlayingButton = null;
            }
            URL.revokeObjectURL(audioUrl); // Clean up the blob URL
        };

        audio.onerror = (error) => {
            // Reset button on error
            if (currentPlayingButton) {
                currentPlayingButton.innerHTML = '▶';
                currentPlayingButton = null;
            }
            log(`ERROR: Audio element error - ${error.message || 'Unknown error'}`);
            URL.revokeObjectURL(audioUrl);
        };
    } catch (error) {
        log(`ERROR: Creating audio failed - ${error.message}`);
    }
}

function audioDataToWAV(audioData, sampleRate) {
    // Convert Float32Array to 16-bit PCM
    const pcmData = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
    }
    return createWAVBuffer(pcmData, sampleRate);
}

function wavToAudioData(wavBuffer, targetSampleRate) {
    // Parse WAV header
    const view = new DataView(wavBuffer.buffer || wavBuffer);
    const sampleRate = view.getUint32(24, true);
    const numChannels = view.getUint16(22, true);
    const bitsPerSample = view.getUint16(34, true);

    // Find data chunk
    let dataOffset = 44; // Default WAV header size
    if (view.getUint32(36, true) === 0x64617461) { // 'data'
        dataOffset = 44;
    }

    // Extract PCM data
    const dataSize = view.getUint32(40, true);
    const numSamples = dataSize / (bitsPerSample / 8) / numChannels;
    const pcmData = new Int16Array(wavBuffer, dataOffset, numSamples);

    // Convert to Float32Array
    const audioData = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
        audioData[i] = pcmData[i] / 32768.0;
    }

    // Resample if needed
    if (sampleRate !== targetSampleRate) {
        const ratio = targetSampleRate / sampleRate;
        const newLength = Math.floor(audioData.length * ratio);
        const resampledData = new Float32Array(newLength);
        for (let i = 0; i < newLength; i++) {
            const index = i / ratio;
            const low = Math.floor(index);
            const high = Math.min(low + 1, audioData.length - 1);
            const weight = index - low;
            resampledData[i] = audioData[low] * (1 - weight) + audioData[high] * weight;
        }
        return resampledData;
    }

    return audioData;
}

function createWAVBuffer(pcmData, sampleRate) {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);

    // RIFF chunk descriptor
    view.setUint8(0, 0x52); // 'R'
    view.setUint8(1, 0x49); // 'I'
    view.setUint8(2, 0x46); // 'F'
    view.setUint8(3, 0x46); // 'F'

    const fileSize = 36 + pcmData.length * 2; // 36 + data size
    view.setUint32(4, fileSize, true); // File size

    // WAVE format
    view.setUint8(8, 0x57); // 'W'
    view.setUint8(9, 0x41); // 'A'
    view.setUint8(10, 0x56); // 'V'
    view.setUint8(11, 0x45); // 'E'

    // Format chunk
    view.setUint8(12, 0x66); // 'f'
    view.setUint8(13, 0x6D); // 'm'
    view.setUint8(14, 0x74); // 't'
    view.setUint8(15, 0x20); // ' '

    view.setUint32(16, 16, true); // Chunk size
    view.setUint16(20, 1, true); // Audio format (PCM)
    view.setUint16(22, 1, true); // Number of channels
    view.setUint32(24, sampleRate, true); // Sample rate
    view.setUint32(28, sampleRate * 2, true); // Byte rate
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // Bits per sample

    // Data chunk
    view.setUint8(36, 0x64); // 'd'
    view.setUint8(37, 0x61); // 'a'
    view.setUint8(38, 0x74); // 't'
    view.setUint8(39, 0x61); // 'a'

    view.setUint32(40, pcmData.length * 2, true); // Data size

    // Combine header and PCM data
    const wavBuffer = new Uint8Array(header.byteLength + pcmData.length * 2);
    wavBuffer.set(new Uint8Array(header), 0);
    wavBuffer.set(new Uint8Array(pcmData.buffer), header.byteLength);

    return wavBuffer;
}

function generateHtmlBlob() {
    const rows = dogLogBody.querySelectorAll('tr:not(.placeholder-row)');
    if (rows.length === 0) {
        return null;
    }

    let htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dog Barking Log - ${new Date().toLocaleDateString()}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .total-recordings { font-size: 18px; color: #666; margin-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        button { padding: 5px 10px; margin: 2px; cursor: pointer; }
        .play-btn { background-color: #007bff; color: white; border: none; border-radius: 3px; }
        .delete-btn { background-color: #dc3545; color: white; border: none; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>Dog Barking Log - ${new Date().toLocaleDateString()}</h1>
    <p>Generated on ${new Date().toLocaleString()}</p>
    <div class="total-recordings">Total Recordings = ${totalRecordings}</div>
    <table>
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Sound Recorded</th>
                <th>Play</th>
            </tr>
        </thead>
        <tbody>
`;

    rows.forEach((row, index) => {
        if (row.audioData && row.sampleRate) {
            // Convert audio data to base64 WAV
            const pcmData = new Int16Array(row.audioData.length);
            for (let i = 0; i < row.audioData.length; i++) {
                pcmData[i] = Math.max(-32768, Math.min(32767, row.audioData[i] * 32768));
            }
            const wavBuffer = createWAVBuffer(pcmData, row.sampleRate);
            const base64Wav = btoa(String.fromCharCode(...wavBuffer));

            htmlContent += `
            <tr>
                <td>${row.timestamp}</td>
                <td>${row.soundName}</td>
                <td><button class="play-btn" onclick="playAudio('data:audio/wav;base64,${base64Wav}')">▶</button></td>
            </tr>
`;
        }
    });

    htmlContent += `
        </tbody>
    </table>
    <div style="margin-top: 30px; padding: 15px; background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; text-align: center; font-size: 14px; color: #666;">
        Generated with <a href="https://dogbarkingdetector.com" target="_blank" style="color: #007bff; text-decoration: none;">dogbarkingdetector.com</a>
    </div>
    <script>
        function playAudio(dataUrl) {
            const audio = new Audio(dataUrl);
            audio.play().catch(err => console.error('Playback error:', err));
        }
    </script>
</body>
</html>
`;

    return new Blob([htmlContent], { type: 'text/html' });
}

function saveAsHtml() {
    const blob = generateHtmlBlob();
    if (!blob) {
        showToast('No dog log entries to save');
        return;
    }

    // Create and download the HTML file
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dog-log-${new Date().toISOString().split('T')[0]}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast('Dog log saved as HTML file');
}

function saveAsCsv() {
    const rows = dogLogBody.querySelectorAll('tr:not(.placeholder-row)');
    if (rows.length === 0) {
        showToast('No dog log entries to save');
        return;
    }

    let csvContent = `Total Recordings = ${totalRecordings}\nTimestamp,Sound Recorded\n`;

    rows.forEach((row) => {
        if (row.timestamp && row.soundName) {
            // Escape commas and quotes in CSV
            const escapedTimestamp = row.timestamp.replace(/"/g, '""');
            const escapedSoundName = row.soundName.replace(/"/g, '""');
            csvContent += `"${escapedTimestamp}","${escapedSoundName}"\n`;
        }
    });

    // Add footer
    csvContent += '\n"Generated with dogbarkingdetector.com","https://dogbarkingdetector.com"';

    // Create and download the CSV file
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dog-log-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast('Dog log saved as CSV file');
}



// Initialize on load
// Accordion functionality
const accordions = document.getElementsByClassName("accordion");

for (let i = 0; i < accordions.length; i++) {
    accordions[i].addEventListener("click", function() {
        this.classList.toggle("active");
        const panel = this.nextElementSibling;
        if (panel.style.display === "block") {
            panel.style.display = "none";
        } else {
            panel.style.display = "block";
        }
    });
}

// Collapsible section functionality
let isCollapsed = false;
const collapsibleTab = document.getElementById('collapsibleTab');
const collapsibleLine = document.getElementById('collapsibleLine');
const infoColumns = document.querySelector('.info-columns');

if (collapsibleTab && infoColumns) {
    collapsibleTab.addEventListener('click', function() {
        isCollapsed = !isCollapsed;
        const tabText = this.querySelector('.tab-text');

        if (isCollapsed) {
            // Hide the columns
            infoColumns.classList.add('collapsed');
            collapsibleLine.classList.add('collapsed');
            tabText.textContent = 'show ▼';
        } else {
            // Show the columns
            infoColumns.classList.remove('collapsed');
            collapsibleLine.classList.remove('collapsed');
            tabText.textContent = 'hide ▲';
        }
    });
}

function handleShareClick() {
    if (hasShared && lastShareUrl) {
        // User has already shared, show the result modal directly
        document.getElementById('shareLinkInput').value = lastShareUrl;
        document.getElementById('shareResultModal').style.display = 'flex';
    } else {
        // First time sharing, show confirmation modal
        showShareConfirmModal();
    }
}

function showShareConfirmModal() {
    const blob = generateHtmlBlob();
    if (!blob) {
        showToast('No dog log entries to share');
        return;
    }

    // Check file size (0x0.st allows up to 512MB, but we'll be conservative)
    if (blob.size > 100 * 1024 * 1024) { // 100MB limit
        showToast('Log file is too large to share (over 100MB)');
        return;
    }

    document.getElementById('shareConfirmModal').style.display = 'flex';
}

function hideShareConfirmModal() {
    document.getElementById('shareConfirmModal').style.display = 'none';

    // Reset progress bar and button states
    const progressContainer = document.getElementById('shareProgressContainer');
    const progressFill = document.getElementById('shareProgressFill');
    const cancelBtn = document.getElementById('shareCancelBtn');
    const confirmBtn = document.getElementById('shareConfirmBtn');

    progressContainer.style.display = 'none';
    progressFill.style.width = '0%';
    cancelBtn.disabled = false;
    confirmBtn.disabled = false;
}

function hideShareResultModal() {
    document.getElementById('shareResultModal').style.display = 'none';
}

async function shareLog() {
    const blob = generateHtmlBlob();
    if (!blob) {
        showToast('No dog log entries to share');
        return;
    }

    // Show progress bar and disable buttons
    const progressContainer = document.getElementById('shareProgressContainer');
    const progressFill = document.getElementById('shareProgressFill');
    const progressText = document.getElementById('shareProgressText');
    const cancelBtn = document.getElementById('shareCancelBtn');
    const confirmBtn = document.getElementById('shareConfirmBtn');

    progressContainer.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Preparing upload...';
    cancelBtn.disabled = true;
    confirmBtn.disabled = true;

    try {
        const array = new Uint8Array(4);
        crypto.getRandomValues(array);
        const shortId = Array.from(array, byte => byte.toString(36)).join('').substring(0, 5);
        const filename = `${shortId}.html`;

        const form = new FormData();
        form.append('file', blob, filename);

        // Update progress
        progressFill.style.width = '25%';
        progressText.textContent = 'Uploading...';

        const response = await fetch('https://dogbark-upload.jatacid.workers.dev/', {
            method: 'POST',
            body: form
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status}`);
        }

        // Update progress to complete
        progressFill.style.width = '100%';
        progressText.textContent = 'Upload complete!';

        const shareUrl = `${window.location.host}/log/${shortId}`;

        // Store share state
        hasShared = true;
        lastShareUrl = shareUrl;

        // Hide confirm modal and show result modal
        hideShareConfirmModal();
        document.getElementById('shareLinkInput').value = shareUrl;
        document.getElementById('shareResultModal').style.display = 'flex';

        showToast('Log uploaded successfully!');

    } catch (error) {
        log(`Share error: ${error.message}`);
        showToast('Upload failed — please check your connection or try again later.');

        // Reset modal state on error
        progressContainer.style.display = 'none';
        cancelBtn.disabled = false;
        confirmBtn.disabled = false;
    }
}

function copyShareLink() {
    const input = document.getElementById('shareLinkInput');
    const urlWithProtocol = 'https://' + input.value;
    input.select();
    input.setSelectionRange(0, 99999); // For mobile devices

    try {
        // Use the URL with protocol for copying
        const tempInput = document.createElement('input');
        tempInput.value = urlWithProtocol;
        document.body.appendChild(tempInput);
        tempInput.select();
        tempInput.setSelectionRange(0, 99999);
        document.execCommand('copy');
        document.body.removeChild(tempInput);
        showToast('Link copied to clipboard!');
    } catch (error) {
        // Fallback for browsers that don't support execCommand
        navigator.clipboard.writeText(urlWithProtocol).then(() => {
            showToast('Link copied to clipboard!');
        }).catch(() => {
            showToast('Failed to copy link');
        });
    }
}

function openShareLink() {
    const input = document.getElementById('shareLinkInput');
    const url = input.value;
    if (url) {
        const fullUrl = 'https://' + url;
        window.open(fullUrl, '_blank');
    }
}

function shareApp() {
    const url = window.location.href;
    const title = 'Dog Barking Detector';
    const text = 'Check out this open source dog barking detector, perfect for monitoring a noisy pet!';

    if (navigator.share) {
        // Use native share API
        navigator.share({
            title: title,
            text: text,
            url: url
        }).catch(err => {
            log(`Share error: ${err.message}`);
        });
    } else {
        // Fallback: copy to clipboard
        const shareText = `${title}\n${text}\n${url}`;
        if (navigator.clipboard) {
            navigator.clipboard.writeText(shareText).then(() => {
                showToast('App link copied to clipboard!');
            }).catch(() => {
                fallbackCopy(shareText);
            });
        } else {
            fallbackCopy(shareText);
        }
    }
}

function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    try {
        document.execCommand('copy');
        showToast('App link copied to clipboard!');
    } catch (err) {
        showToast('Unable to copy link. Please copy manually.');
    }
    document.body.removeChild(textArea);
}

function handleDragOver(event) {
    event.preventDefault();
    dropZone.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    dropZone.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    dropZone.classList.remove('dragover');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        // Create a fake event to reuse handleFileSelect
        const fakeEvent = { target: { files: [files[0]] } };
        handleFileSelect(fakeEvent);
    }
}

function cancelProcessing() {
    if (isProcessingFile) {
        isProcessingFile = false;
        showToast('File processing cancelled');

        // Reset UI
        fileProgress.style.display = 'none';
        processFileBtn.disabled = false;
        processFileBtn.textContent = 'Start Processing';
        cancelProcessBtn.style.display = 'none';
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) {
        selectedFile = null;
        fileInfo.style.display = 'none';
        processFileBtn.disabled = true;
        return;
    }

    // Validate file type
    const validTypes = ['audio/', 'video/'];
    const isValidType = validTypes.some(type => file.type.startsWith(type));

    if (!isValidType) {
        showToast('Please select a valid audio or video file.');
        selectedFile = null;
        fileInfo.style.display = 'none';
        processFileBtn.disabled = true;
        return;
    }

    // Check file size (limit to 100MB for browser compatibility)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
        showToast('File is too large. Please select a file smaller than 100MB.');
        selectedFile = null;
        fileInfo.style.display = 'none';
        processFileBtn.disabled = true;
        return;
    }

    selectedFile = file;

    // Show file info
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    const duration = file.duration ? `${file.duration.toFixed(1)}s` : 'Unknown duration';
    fileInfo.innerHTML = `<strong>${file.name}</strong><br>Size: ${sizeMB} MB<br>Type: ${file.type}`;
    fileInfo.style.display = 'block';

    // Enable process button if file models are loaded
    processFileBtn.disabled = !isFileModelsLoaded;
    if (!isFileModelsLoaded) {
        processFileBtn.textContent = 'Load models first';
    } else {
        processFileBtn.textContent = 'Start Processing';
    }
}

async function processFile() {
    if (!selectedFile || !isFileModelsLoaded) {
        showToast('Please select a file and load models first.');
        return;
    }

    if (isProcessingFile) {
        showToast('Already processing a file.');
        return;
    }

    try {
        isProcessingFile = true;
        processFileBtn.disabled = true;
        processFileBtn.textContent = 'Processing...';
        cancelProcessBtn.style.display = 'inline-block';

        // Show progress
        fileProgress.style.display = 'block';
        updateFileProgress(0, 'Decoding audio...');

        // Create audio context for decoding
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Read file as array buffer
        const arrayBuffer = await selectedFile.arrayBuffer();

        // Decode audio data
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        updateFileProgress(10, 'Audio decoded, analyzing...');

        // Get mono audio data (mix channels if stereo)
        const channelData = audioBuffer.getChannelData(0); // Use first channel
        const sampleRate = audioBuffer.sampleRate;
        const totalSamples = channelData.length;

        log(`Processing file: ${selectedFile.name}, duration: ${(totalSamples / sampleRate).toFixed(1)}s, sample rate: ${sampleRate}Hz`);

        // Process audio in chunks
        await processAudioFile(channelData, sampleRate, totalSamples);

        // Hide progress
        fileProgress.style.display = 'none';

        // Reset UI
        processFileBtn.disabled = false;
        processFileBtn.textContent = 'Start Processing';
        cancelProcessBtn.style.display = 'none';
        isProcessingFile = false;

        showToast('File processing complete!');

    } catch (error) {
        log(`File processing error: ${error.message}`);
        showToast(`Error processing file: ${error.message}`);

        // Reset UI
        fileProgress.style.display = 'none';
        processFileBtn.disabled = false;
        processFileBtn.textContent = 'Start Processing';
        cancelProcessBtn.style.display = 'none';
        isProcessingFile = false;
    }
}

function updateFileProgress(percentage, text) {
    fileProgressFill.style.width = percentage + '%';
    progressText.textContent = text;
}

async function processAudioFile(audioData, sampleRate, totalSamples) {
    // Reset buffers for file processing
    windowBuffer = new Float32Array(WINDOW_SIZE);
    saveBuffer = new Float32Array(SAVE_BUFFER_SIZE);
    saveIndex = 0;
    bufferFilled = false;
    windowPosition = 0;
    predictionBuffer = [];
    predictionCount = 0;
    fileCurrentTime = 0;

    // Resample to 16kHz if needed
    let processedData = audioData;
    if (sampleRate !== 16000) {
        processedData = resampleAudio(audioData, sampleRate, 16000);
        sampleRate = 16000;
    }

    const chunkSize = 10000; // Process in 10k sample chunks (about 0.625s at 16kHz)
    let processedSamples = 0;

    for (let i = 0; i < processedData.length; i += chunkSize) {
        if (!isProcessingFile) break; // Allow cancellation

        const chunk = processedData.slice(i, i + chunkSize);
        processedSamples += chunk.length;

        // Update current time position
        fileCurrentTime = processedSamples / sampleRate;

        // Update progress
        const progress = Math.min(90, 10 + (processedSamples / processedData.length) * 80);
        updateFileProgress(progress, `Analyzing... ${Math.round(fileCurrentTime * 10) / 10}s processed`);

        // Process chunk
        await processAudioChunk(chunk, sampleRate);

        // Small delay to prevent blocking UI
        await new Promise(resolve => setTimeout(resolve, 1));
    }

    updateFileProgress(100, 'Analysis complete');
}

function initModalListeners() {
    const shareCancelBtn = document.getElementById('shareCancelBtn');
    const shareConfirmBtn = document.getElementById('shareConfirmBtn');
    const shareCloseBtn = document.getElementById('shareCloseBtn');
    const copyLinkBtn = document.getElementById('copyLinkBtn');
    const openLinkBtn = document.getElementById('openLinkBtn');

    if (shareCancelBtn) shareCancelBtn.addEventListener('click', hideShareConfirmModal);
    if (shareConfirmBtn) shareConfirmBtn.addEventListener('click', shareLog);
    if (shareCloseBtn) shareCloseBtn.addEventListener('click', hideShareResultModal);
    if (copyLinkBtn) copyLinkBtn.addEventListener('click', copyShareLink);
    if (openLinkBtn) openLinkBtn.addEventListener('click', openShareLink);

    // Close modals when clicking outside
    const confirmModal = document.getElementById('shareConfirmModal');
    const resultModal = document.getElementById('shareResultModal');
    if (confirmModal) {
        confirmModal.addEventListener('click', function(e) {
            if (e.target === this) {
                // Don't allow closing if upload is in progress
                const progressContainer = document.getElementById('shareProgressContainer');
                if (progressContainer && progressContainer.style.display !== 'none') {
                    return;
                }
                hideShareConfirmModal();
            }
        });
    }
    if (resultModal) {
        resultModal.addEventListener('click', function(e) {
            if (e.target === this) hideShareResultModal();
        });
    }
}

window.addEventListener('load', init);
