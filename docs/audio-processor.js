class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input && input[0]) {
            const inputBuffer = input[0];
            // Calculate amplitude stats for debugging
            const avgAmplitude = inputBuffer.reduce((sum, val) => sum + Math.abs(val), 0) / inputBuffer.length;
            const maxAmplitude = Math.max(...inputBuffer.map(Math.abs));
            // Send the audio data to the main thread
            this.port.postMessage({
                type: 'audioData',
                data: inputBuffer.slice(), // Copy the buffer
                stats: {
                    avgAmplitude: avgAmplitude,
                    maxAmplitude: maxAmplitude,
                    sampleCount: inputBuffer.length
                }
            });
        }
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);