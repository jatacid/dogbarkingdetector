let model;
let audioContext;
let analyser;
let microphone;
let processor;
let isRecording = false;
let startStopBtn;
let canvas;
let canvasContext;
let animationId;
let sensitivitySlider;
let sensitivityValue;
let detectionList;
let dogLogBody;
let sensitivity = 0.3;
let lastDetectionTime = 0;
let modelLoaded = false;
let debugMode = false;
let debugLogElement;
let audioChunkCounter = 0;
let mediaStream = null;

async function init() {
    startStopBtn = document.getElementById('startStopBtn');
    canvas = document.getElementById('waveform');
    canvasContext = canvas.getContext('2d');
    sensitivitySlider = document.getElementById('sensitivity');
    sensitivityValue = document.getElementById('sensitivityValue');
    detectionList = document.getElementById('detectionList');
    dogLogBody = document.getElementById('dogLogBody');

    // Check for debug mode
    const urlParams = new URLSearchParams(window.location.search);
    debugMode = urlParams.has('debug');
    if (debugMode) {
        debugLogElement = document.getElementById('debugLog');
        document.getElementById('debugPanel').style.display = 'block';
        debugLog('=== DEBUG MODE STARTED ===');
        debugLog(`User Agent: ${navigator.userAgent}`);
        debugLog(`Platform: ${navigator.platform}`);
        debugLog(`Audio Context Support: ${!!window.AudioContext || !!window.webkitAudioContext}`);
        debugLog(`Media Devices Support: ${!!navigator.mediaDevices && !!navigator.mediaDevices.getUserMedia}`);
    }

    startStopBtn.addEventListener('click', toggleRecording);
    sensitivitySlider.addEventListener('input', updateSensitivity);

    // Add placeholder entries to dog log immediately
    addPlaceholderEntries();

    // Initialize detection list with placeholder values
    updateDetectionList([]);
}

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        // Disable button and show loading state
        startStopBtn.disabled = true;
        startStopBtn.classList.add('loading');
        startStopBtn.textContent = 'Initializing...';

        // Load TensorFlow and model only when starting recording (if not already loaded)
        if (!modelLoaded) {
            await tf.ready();
            startStopBtn.textContent = 'Loading model...';
            try {
                model = await yamnet.load('./model/');
                modelLoaded = true;
            } catch (error) {
                if (debugMode) {
                    debugLog('ERROR: Failed to load model: ' + error.message);
                }
                console.error('ERROR: Failed to load model: ' + error.message);
                startStopBtn.disabled = false;
                startStopBtn.classList.remove('loading');
                startStopBtn.textContent = 'Start Recording';
                return;
            }
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });
        mediaStream = stream; // Store reference to stop tracks later

        startStopBtn.textContent = 'Setting up audio...';
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            window.audioContextInstance = audioContext; // Store globally for reuse in playback
            if (debugMode) {
                debugLog(`AudioContext created: sampleRate=${audioContext.sampleRate}Hz, state=${audioContext.state}`);
                debugLog(`MediaStream: ${stream.getAudioTracks().length} audio track(s)`);
                const track = stream.getAudioTracks()[0];
                if (track) {
                    debugLog(`Audio Track: enabled=${track.enabled}, muted=${track.muted}, readyState=${track.readyState}`);
                    const settings = track.getSettings();
                    debugLog(`Track Settings: sampleRate=${settings.sampleRate}, channelCount=${settings.channelCount}, echoCancellation=${settings.echoCancellation}, noiseSuppression=${settings.noiseSuppression}, autoGainControl=${settings.autoGainControl}`);
                }
            }
            console.log(`AudioContext sample rate: ${audioContext.sampleRate}Hz`);
            microphone = audioContext.createMediaStreamSource(stream);
        } catch (audioError) {
            console.error('AudioContext error: ' + audioError.message);
            if (debugMode) {
                debugLog(`AudioContext error: ${audioError.message}`);
            }
            console.log(`AudioContext error: ${audioError.message}`);
            startStopBtn.disabled = false;
            startStopBtn.classList.remove('loading');
            startStopBtn.textContent = 'Start Recording';
            showToast('Audio setup failed: ' + audioError.message);
            return;
        }

        // Create analyser for waveform visualization
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0; // No smoothing for raw data
        microphone.connect(analyser);

        // Load and create AudioWorklet for raw audio capture
        await audioContext.audioWorklet.addModule('audio-processor.js');
        processor = new AudioWorkletNode(audioContext, 'audio-processor');

        // Create analyser for waveform visualization
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        microphone.connect(analyser);

        microphone.connect(processor);
        if (!debugMode) {
            processor.connect(audioContext.destination);
        }

        let audioBuffer = [];

        processor.port.onmessage = (event) => {
            if (event.data.type === 'audioData') {
                const inputBuffer = event.data.data;
                const stats = event.data.stats;
                audioBuffer.push(...inputBuffer);

                // Remove AudioWorklet logging to reduce spam

                // Process in 1s chunks (YAMNet standard)
                const sampleRate = audioContext.sampleRate;
                const chunkSize = sampleRate * 1; // 1s
                if (audioBuffer.length >= chunkSize) {
                    const chunk = audioBuffer.splice(0, chunkSize);
                    processAudio(chunk, sampleRate);
                }
            }
        };

        isRecording = true;

        // Show hidden sections (they are now always visible)
        document.getElementById('audioRecording').style.display = 'block';
        document.getElementById('audioPlaceholder').style.display = 'none';
        document.getElementById('waveform').style.display = 'block';
        document.getElementById('detection').style.display = 'block';
        document.getElementById('detectionPlaceholder').style.display = 'none';
        document.getElementById('dogLog').style.display = 'block';

        // Start waveform animation
        drawWaveform();

    } catch (error) {
        if (debugMode) {
            debugLog('Error accessing microphone: ' + error.message);
        }
        console.error('Error accessing microphone: ' + error.message);
        // Handle permission denied gracefully
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            showToast('Microphone access denied. Please allow microphone access and try again.');
            startStopBtn.disabled = false;
            startStopBtn.classList.remove('loading');
            startStopBtn.textContent = 'Request Permission & Start Recording';
        } else {
            // Reset button state on other errors
            startStopBtn.disabled = false;
            startStopBtn.classList.remove('loading');
            startStopBtn.textContent = 'Start Recording';
        }
    }
}

function stopRecording() {
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (analyser) {
        analyser.disconnect();
        analyser = null;
    }
    if (microphone) {
        microphone.disconnect();
        microphone = null;
    }
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }

    // Stop all media tracks to turn off browser recording indicator
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => {
            track.stop();
        });
        mediaStream = null;
    }

    // Stop the audio context to fully release microphone
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().then(() => {
            console.log('AudioContext closed successfully');
            audioContext = null;
        }).catch(err => {
            console.error('Error closing AudioContext:', err);
            audioContext = null;
        });
    }

    isRecording = false;
    startStopBtn.disabled = false;
    startStopBtn.classList.remove('loading');
    startStopBtn.classList.remove('stop-btn');
    startStopBtn.textContent = 'Start Recording';

    // Keep sections visible except waveform
    document.getElementById('waveform').style.display = 'none';
    document.getElementById('audioPlaceholder').style.display = 'block';
    document.getElementById('detectionPlaceholder').style.display = 'block';

    // Clear canvas
    canvasContext.fillStyle = '#f8f8f8';
    canvasContext.fillRect(0, 0, canvas.width, canvas.height);
}

async function processAudio(audioData, sampleRate) {
    const timestamp = new Date().toLocaleTimeString();

    audioChunkCounter++;
    if (debugMode && audioChunkCounter % 3 === 0) { // Log every 3rd chunk
        const rawAvg = audioData.reduce((sum, val) => sum + Math.abs(val), 0) / audioData.length;
        const rawMax = Math.max(...audioData.map(Math.abs));
        debugLog(`=== AUDIO CHUNK ${audioChunkCounter} ===`);
        debugLog(`Raw captured audio: ${audioData.length} samples at ${sampleRate}Hz, avg=${rawAvg.toFixed(6)}, max=${rawMax.toFixed(6)}`);
    }

    // Resample to 16kHz if necessary
    let resampledData = audioData;
    if (sampleRate !== 16000) {
        const ratio = 16000 / sampleRate;
        const newLength = Math.floor(audioData.length * ratio);
        resampledData = new Float32Array(newLength);
        for (let i = 0; i < newLength; i++) {
            const index = i / ratio;
            const low = Math.floor(index);
            const high = Math.min(low + 1, audioData.length - 1);
            const weight = index - low;
            resampledData[i] = audioData[low] * (1 - weight) + audioData[high] * weight;
        }
        if (debugMode && audioChunkCounter % 3 === 0) {
            const resampledAvg = resampledData.reduce((sum, val) => sum + Math.abs(val), 0) / resampledData.length;
            const resampledMax = Math.max(...resampledData.map(Math.abs));
            debugLog(`After resampling: ${newLength} samples at 16kHz, avg=${resampledAvg.toFixed(6)}, max=${resampledMax.toFixed(6)}`);
        }
    }

    // YAMNet expects exactly 16000 samples (1 second at 16kHz)
    const yamnetInput = new Float32Array(16000);
    const copyLength = Math.min(resampledData.length, 16000);
    yamnetInput.set(resampledData.slice(0, copyLength));

    // Apply normalization to counteract mobile audio processing differences
    // Method 1: RMS normalization (preserve dynamics but normalize overall level)
    const rms = Math.sqrt(yamnetInput.reduce((sum, val) => sum + val * val, 0) / yamnetInput.length);
    const targetRMS = 0.1; // Target RMS level (adjustable)
    if (rms > 0) {
        const scale = targetRMS / rms;
        for (let i = 0; i < yamnetInput.length; i++) {
            yamnetInput[i] *= scale;
        }
    }

    // Method 2: Remove DC offset more aggressively
    const dcOffset = yamnetInput.reduce((sum, val) => sum + val, 0) / yamnetInput.length;
    for (let i = 0; i < yamnetInput.length; i++) {
        yamnetInput[i] -= dcOffset;
    }

    // Method 3: Apply a slight high-pass filter to remove low-frequency noise
    const alpha = 0.95; // High-pass filter coefficient
    let prevFiltered = 0;
    for (let i = 0; i < yamnetInput.length; i++) {
        const filtered = alpha * (prevFiltered + yamnetInput[i] - yamnetInput[i > 0 ? i - 1 : 0]);
        yamnetInput[i] = filtered;
        prevFiltered = filtered;
    }

    // Log YAMNet input characteristics after normalization
    const yamnetAvg = yamnetInput.reduce((sum, val) => sum + Math.abs(val), 0) / yamnetInput.length;
    const yamnetDc = yamnetInput.reduce((sum, val) => sum + val, 0) / yamnetInput.length;
    const yamnetMax = Math.max(...yamnetInput.map(Math.abs));
    const yamnetRMS = Math.sqrt(yamnetInput.reduce((sum, val) => sum + val * val, 0) / yamnetInput.length);
    if (debugMode && audioChunkCounter % 3 === 0) {
        debugLog(`After normalization: avg=${yamnetAvg.toFixed(6)}, max=${yamnetMax.toFixed(6)}, rms=${yamnetRMS.toFixed(6)}, dc=${yamnetDc.toFixed(6)}`);
    }
    console.log(`YAMNet input: avg amplitude=${yamnetAvg.toFixed(6)}, max amplitude=${yamnetMax.toFixed(6)}, dc offset=${yamnetDc.toFixed(6)}, copyLength=${copyLength}`);

    try {
        // Use the yamnet.js wrapper's predict method (handles input shaping correctly)
        const predictions = await model.predict(yamnetInput);
        const scores = await predictions.data();

        // Find top 3 detected classes (limit to 521 classes as per YAMNet spec)
        const topClasses = [];
        const maxClasses = Math.min(scores.length, 521); // Ensure we don't exceed 521 classes
        for (let i = 0; i < maxClasses; i++) {
            topClasses.push({ index: i, score: scores[i], name: model.classNames[i] || `Class ${i}` });
        }
        topClasses.sort((a, b) => b.score - a.score);

        // Log top 5 classes
        const top5Debug = topClasses.slice(0, 5).map(c => `${c.name}: ${(c.score * 100).toFixed(1)}%`).join(', ');
        console.log(`Top 5 classes: ${top5Debug}`);

        // Check dog-related classes (dog, bark, yip, howl, bow-wow, growling, whimper, domestic animals pets)
        const dogSoundClasses = [69, 70, 71, 72, 73, 74, 75, 237]; // Dog, Bark, Yip, Howl, Bow-wow, Growling, Whimper, Domestic animals pets
        const dogSoundScores = dogSoundClasses.map(classIndex => ({
            class: classIndex,
            score: scores[classIndex],
            name: model.classNames[classIndex] || `Class ${classIndex}`
        }));

        // Log dog sound scores
        const dogScoresDebug = dogSoundScores.map(s => `${s.name}: ${(s.score * 100).toFixed(1)}%`).join(', ');
        console.log(`Dog sound scores: ${dogScoresDebug}`);

        if (debugMode && audioChunkCounter % 3 === 0) {
            debugLog(`ML Results: Top classes - ${top5Debug}`);
            debugLog(`ML Results: Dog sounds - ${dogScoresDebug}`);
            debugLog(`ML Results: Inference completed successfully`);
        }

        // Separate specific and generic dog sounds
        const specificClasses = [70, 71, 72, 73, 74, 75]; // Bark, Yip, Howl, Bow-wow, Growling, Whimper
        const genericClasses = [69]; // Dog
        const specificScores = dogSoundScores.filter(s => specificClasses.includes(s.class));
        const genericScores = dogSoundScores.filter(s => genericClasses.includes(s.class));

        // Prioritize specific dog vocalizations over generic classifications
        let bestDogSound = null;
        if (specificScores.length > 0) {
            const bestSpecific = specificScores.reduce((best, current) =>
                current.score > best.score ? current : best
            );
            if (bestSpecific.score > sensitivity) {
                bestDogSound = bestSpecific;
            }
        }
        if (!bestDogSound && genericScores.length > 0) {
            const bestGeneric = genericScores.reduce((best, current) =>
                current.score > best.score ? current : best
            );
            if (bestGeneric.score > sensitivity) {
                bestDogSound = bestGeneric;
            }
        }

        // Update detection list with top classes
        updateDetectionList(topClasses);

        // Show what the model is detecting
        const top3 = topClasses.slice(0, 3);
        const detectionInfo = top3.map(c => `${c.name}: ${(c.score * 100).toFixed(1)}%`).join(', ');

        // Use sensitivity slider value for dog-related sound detection threshold
        // Collect all dog-related sounds above sensitivity threshold
        const detectedDogSounds = dogSoundScores.filter(sound => sound.score > sensitivity);

        if (detectedDogSounds.length > 0) {
            const now = Date.now();
            if (now - lastDetectionTime >= 4000) { // 4 second debounce
                // Format all detected sounds as comma-delimited list: "class (percentage), class (percentage), ..."
                const soundList = detectedDogSounds
                    .map(sound => `${sound.name} (${(sound.score * 100).toFixed(1)}%)`)
                    .join(', ');

                if (debugMode && audioChunkCounter % 3 === 0) {
                    debugLog(`DETECTION: Dog sounds detected - ${soundList}`);
                }
                console.log(`Dog detection triggered: ${soundList}`);
                showToast('🐕 Dog detected!');
                addToDogLog(timestamp, soundList, '', new Float32Array(audioData), sampleRate);
                lastDetectionTime = now;
            }
        } else {
            if (debugMode && audioChunkCounter % 3 === 0) {
                debugLog('DETECTION: No dog sounds detected above threshold');
            }
            console.log('No dog sounds detected above threshold');
        }

        predictions.dispose();

    } catch (error) {
        if (debugMode && audioChunkCounter % 3 === 0) {
            debugLog(`ERROR: Inference failed: ${error.message}`);
        }
        console.error(`ERROR: Inference failed: ${error.message}`);
    }
}

function drawWaveform() {
    if (!isRecording) return;

    animationId = requestAnimationFrame(drawWaveform);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);

    // Clear canvas
    canvasContext.fillStyle = '#f8f8f8';
    canvasContext.fillRect(0, 0, canvas.width, canvas.height);

    // Draw waveform
    canvasContext.lineWidth = 2;
    canvasContext.strokeStyle = '#007bff';
    canvasContext.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;

        if (i === 0) {
            canvasContext.moveTo(x, y);
        } else {
            canvasContext.lineTo(x, y);
        }

        x += sliceWidth;
    }

    canvasContext.lineTo(canvas.width, canvas.height / 2);
    canvasContext.stroke();
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
        topClasses.slice(0, 5).forEach((classInfo, index) => {
            const detectionItem = document.createElement('div');
            detectionItem.className = 'detection-item';
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

    const formattedTimestamp = `${day}${getOrdinalSuffix(day)} ${month} ${year}, ${displayHours}:${minutes}${ampm}`;

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

    dogLogBody.appendChild(row);
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

function debugLog(message) {
    if (!debugMode || !debugLogElement) return;

    const logEntry = document.createElement('div');
    const timestamp = new Date().toLocaleTimeString();
    logEntry.textContent = `[${timestamp}] ${message}`;
    debugLogElement.appendChild(logEntry);

    // Auto-scroll to bottom
    debugLogElement.scrollTop = debugLogElement.scrollHeight;
}




async function playAudio(audioData, sampleRate) {
    try {
        // Convert Float32Array to 16-bit PCM for WAV format
        const pcmData = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            // Convert from -1.0 to 1.0 range to -32768 to 32767
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
                console.error('Error starting audio playback: ' + err.message);
            });
        };

        audio.onended = () => {
            URL.revokeObjectURL(audioUrl); // Clean up the blob URL
        };

        audio.onerror = (error) => {
            console.error('Audio element error: ' + error.message);
            URL.revokeObjectURL(audioUrl);
        };
    } catch (error) {
        console.error('Error creating audio: ' + error.message);
    }
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
window.addEventListener('load', init);
