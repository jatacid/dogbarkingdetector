let model;
let audioContext;
let microphone;
let processor;
let isRecording = false;
let isLoaded = false;
let startStopBtn;
let loadBtn;
let animationId;
let sensitivitySlider;
let sensitivityValue;
let detectionList;
let dogLogBody;
let sensitivity = 0.3;
let lastDetectionTime = 0;
let lastLoggedDetectionTime = 0;
let mediaStream = null;
let saveAsHtmlBtn;
let saveAsCsvBtn;
let audioTickCount = 0;
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
    sensitivitySlider = document.getElementById('sensitivity');
    sensitivityValue = document.getElementById('sensitivityValue');
    detectionList = document.getElementById('detectionList');
    dogLogBody = document.getElementById('dogLogBody');
    saveAsHtmlBtn = document.getElementById('saveAsHtmlBtn');
    saveAsCsvBtn = document.getElementById('saveAsCsvBtn');

    loadBtn.addEventListener('click', loadModels);
    startStopBtn.addEventListener('click', toggleRecording);
    sensitivitySlider.addEventListener('input', updateSensitivity);
    saveAsHtmlBtn.addEventListener('click', saveAsHtml);
    saveAsCsvBtn.addEventListener('click', saveAsCsv);

    // Add placeholder entries to dog log immediately
    addPlaceholderEntries();

    // Initialize detection list with placeholder values
    updateDetectionList([]);
}

async function loadModels() {
    try {
        loadBtn.disabled = true;
        loadBtn.classList.add('loading');
        loadBtn.querySelector('.button-text').textContent = 'Initializing...';
        updateProgressBar(10);

        log('Initializing...');

        // Set WASM backend explicitly for consistent performance across devices
        await tf.setBackend('wasm');
        await tf.ready();
        log('WASM backend ready');
        loadBtn.querySelector('.button-text').textContent = 'Setting up TensorFlow...';
        updateProgressBar(25);

        loadBtn.querySelector('.button-text').textContent = 'Loading YAMNet model...';
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
        updateProgressBar(90);
        // Warm up YAMNet inference engine with a test prediction
        const testBuffer = new Float32Array(16000).fill(0);
        const warmupPredictions = await model.predict(testBuffer);
        warmupPredictions.dispose();

        // Everything loaded successfully
        isLoaded = true;
        loadBtn.classList.remove('loading');
        loadBtn.querySelector('.button-text').textContent = 'Loaded ✓';
        loadBtn.disabled = true;
        startStopBtn.disabled = false;
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

    
    // Already logged above
}

function processAudioChunk(audioData, sampleRate) {
    try {
        // Add data to window buffer (already resampled to 16kHz)
        for (let i = 0; i < audioData.length; i++) {
            if (windowPosition < WINDOW_SIZE) {
                windowBuffer[windowPosition] = audioData[i];
                windowPosition++;
            } else {
                // Window is full, shift by hop size and add new data
                windowBuffer.copyWithin(0, HOP_SIZE);
                windowPosition = WINDOW_SIZE - HOP_SIZE;
                for (let j = 0; j < Math.min(HOP_SIZE, audioData.length - i); j++) {
                    windowBuffer[windowPosition + j] = audioData[i + j];
                }
                windowPosition += Math.min(HOP_SIZE, audioData.length - i);
                i += Math.min(HOP_SIZE, audioData.length - i) - 1;

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
                const currentTime = Date.now() / 1000; // Current time in seconds
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

                if (timeSinceLastLog >= 4) {
                    // Log to console and add to dog log
                    log(`Detection: ${soundList}`);
                    showToast('Detection logged!');

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

                    const timestamp = new Date().toLocaleTimeString();
                    addToDogLog(timestamp, soundList, '', capturedAudio, 16000);
                    lastLoggedDetectionTime = currentTime;
                } else {
                    // Log to console with cooldown message, but don't add to dog log
                    log(`Detection Within 4 Second Cooldown: ${soundList}`);
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
}

function log(message) {
    console.log(`[DogBarkDetector] ${message}`);
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
                // Playback started successfully
            }).catch(err => {
                log(`ERROR: Audio playback failed - ${err.message}`);
            });
        };

        audio.onended = () => {
            URL.revokeObjectURL(audioUrl); // Clean up the blob URL
        };

        audio.onerror = (error) => {
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

function saveAsHtml() {
    const rows = dogLogBody.querySelectorAll('tr:not(.placeholder-row)');
    if (rows.length === 0) {
        showToast('No dog log entries to save');
        return;
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

    // Create and download the HTML file
    const blob = new Blob([htmlContent], { type: 'text/html' });
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

    let csvContent = 'Timestamp,Sound Recorded\n';

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

window.addEventListener('load', init);
