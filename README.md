# Dog Barking Detector

**Live Demo:** <a href="https://jatacid.github.io/dogbarkingdetector" target="_blank">https://jatacid.github.io/dogbarkingdetector</a>
**Domain:** <a href="https://dogbarkingdetector.com/" target="_blank">https://dogbarkingdetector.com/</a>

A browser-based, offline dog barking detection tool that uses machine learning to identify and log dog vocalizations in real-time. Built with TensorFlow.js and the YAMNet audio classification model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Static Files

All application files are located in the `/docs` folder:

## Project Structure

```
docs/
├── app.js                    # Main application logic
├── audio-processor.js        # AudioWorklet for real-time audio capture
├── index.html               # Main HTML interface
├── style.css                # CSS styling
├── favicon.png              # App icon
├── privacy-policy.html      # Privacy policy page
├── robots.txt               # SEO configuration
├── sitemap.xml              # SEO sitemap
├── libs/                    # Third-party libraries
│   └── yamnet.js                  # YAMNet wrapper
└── model/                   # YAMNet machine learning model
    ├── model.json           # Model configuration
    ├── group1-shard1of4.bin # Model weights (sharded)
    ├── group1-shard2of4.bin
    ├── group1-shard3of4.bin
    ├── group1-shard4of4.bin
    ├── assets/
    │   └── yamnet_class_map.csv # Class labels
    └── variables/
        ├── variables.data-00000-of-00001
        └── variables.index
```


## How It Works

### Core Functionality

1. **Audio Capture**: Uses Web Audio API to access microphone input
2. **Real-time Processing**: AudioWorklet captures raw audio data in real-time
3. **Machine Learning**: YAMNet model classifies 1-second audio chunks into 521 sound categories
4. **Dog Detection**: Specifically monitors for dog vocalization classes such as Dog, Bark, YIp, etc
5. **Logging**: Records detections with timestamps and audio playback capability
6. **Visualization**: Real-time detection confidence scores


### Technical Details

- **Model**: YAMNet (converted to TensorFlow.js format)
- **Sample Rate**: 16kHz (resampled from browser's native rate)
- **Chunk Size**: 1 second of audio per inference
- **Classes**: 521 total, focused on dog sounds (classes 70-75)
- **Privacy**: All processing happens client-side, no audio data leaves the browser
- **Dependencies**: TensorFlow.js (loaded from CDN), YAMNet wrapper library

## Important Notes

### Legal Disclaimer
This tool is intended for personal use only. Users are responsible for complying with all applicable local laws and regulations regarding audio recording and privacy.

### Performance Requirements
The application requires modern browsers with Web Audio API support and sufficient hardware capabilities for real-time machine learning inference. Performance may vary on lower-end devices.

### Detection Algorithm

1. Capture 1-second audio chunks
2. Resample to 16kHz if necessary
3. Run inference through YAMNet model
4. Check scores for dog-related classes above sensitivity threshold
5. Prioritize specific vocalizations (bark, yip, howl) over generic
6. Log detection with timestamp and audio data
7. Debounce detections to prevent spam (4-second minimum interval)


### Browser Compatibility

Requires modern browser with:
- Web Audio API support
- AudioWorklet support
- getUserMedia for microphone access
- TensorFlow.js compatible browser

Tested on Chrome, Firefox, Safari, and Edge.
