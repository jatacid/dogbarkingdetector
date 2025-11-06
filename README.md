# Dog Barking Detector

**Live Demo:** https://jatacid.github.io/dogbarkingdetector
**Domain:** http://dogbarkingdetector.com/

A browser-based, offline dog barking detection tool that uses machine learning to identify and log dog vocalizations in real-time. Built with TensorFlow.js and the YAMNet audio classification model.

## Static Files

All application files are located in the `/dogbarkingdetector` folder:

## Project Structure

```
dogbarkingdetector/
├── app.js                    # Main application logic
├── audio-processor.js        # AudioWorklet for real-time audio capture
├── index.html               # Main HTML interface
├── style.css                # CSS styling
├── favicon.png              # App icon
├── robots.txt               # SEO configuration
├── sitemap.xml              # SEO sitemap
├── libs/                    # Third-party libraries
│   ├── tf.min.js            # TensorFlow.js
│   ├── yamnet.js            # YAMNet wrapper
│   └── yamnet.min.js        # Minified YAMNet wrapper
└── model/                   # YAMNet machine learning model
    ├── model.json           # Model configuration
    ├── group1-shard1of4.bin # Model weights (sharded)
    ├── group1-shard2of4.bin
    ├── group1-shard3of4.bin
    ├── group1-shard4of4.bin
    ├── assets/
    │   └── yamnet_class_map.csv # Class labels
    └── variables/
        └── variables.data-00000-of-00001
        └── variables.index
```


## How It Works

### Core Functionality

1. **Audio Capture**: Uses Web Audio API to access microphone input
2. **Real-time Processing**: AudioWorklet captures raw audio data in real-time
3. **Machine Learning**: YAMNet model classifies 1-second audio chunks into 521 sound categories
4. **Dog Detection**: Specifically monitors for dog vocalizations (classes 69-75: Dog, Bark, Yip, Howl, Bow-wow, Growling, Whimper)
5. **Logging**: Records detections with timestamps and audio playback capability
6. **Visualization**: Real-time waveform display and detection confidence scores

### Key Components

#### app.js
- Initializes the application and UI elements
- Manages microphone access and AudioContext setup
- Loads TensorFlow.js and YAMNet model on demand
- Processes audio in 1-second chunks at 16kHz
- Handles detection logic with configurable sensitivity threshold
- Updates UI with detection results and waveform visualization
- Manages audio playback of recorded clips

#### audio-processor.js
- AudioWorkletProcessor that runs in a separate thread
- Captures raw audio data from microphone input
- Sends audio buffers to main thread for processing

#### index.html
- Single-page application interface
- Contains controls for starting/stopping recording
- Displays detection results, waveform, and log table
- Includes sensitivity slider for adjusting detection threshold
- Optional debug mode (enable with ?debug URL parameter)

### Debug Mode

Debug mode provides detailed logging and additional information for troubleshooting and development:

- **Enable**: Add `?debug` to the URL (e.g., `index.html?debug`)
- **Features**:
  - Debug panel with real-time log output which only appears if the ?debug is selected. And activates alternative code, sort of like a beta version to allow testing of changes.
  - Whenever making changes that are a little experimental - try putting them behind the ?debug mode first

### Technical Details

- **Model**: YAMNet (converted to TensorFlow.js format)
- **Sample Rate**: 16kHz (resampled from browser's native rate)
- **Chunk Size**: 1 second of audio per inference
- **Classes**: 521 total, focused on dog sounds (classes 70-75)
- **Privacy**: All processing happens client-side, no audio data leaves the browser
- **Dependencies**: TensorFlow.js, YAMNet wrapper library

### Detection Algorithm

1. Capture 1-second audio chunks
2. Resample to 16kHz if necessary
3. Run inference through YAMNet model
4. Check scores for dog-related classes above sensitivity threshold
5. Prioritize specific vocalizations (bark, yip, howl) over generic "dog" class
6. Log detection with timestamp and audio data
7. Debounce detections to prevent spam (4-second minimum interval)


### Browser Compatibility

Requires modern browser with:
- Web Audio API support
- AudioWorklet support
- getUserMedia for microphone access
- TensorFlow.js compatible browser

Tested on Chrome, Firefox, Safari, and Edge.