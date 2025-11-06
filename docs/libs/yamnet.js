/**
 * YAMNet Audio Classification Model Wrapper
 * This wrapper provides a simple interface to load and use the YAMNet model for audio classification
 */

class YAMNet {
    constructor() {
        this.model = null;
        this.classNames = null;
    }

    /**
     * Load the YAMNet model
     * @param {string} modelPath - Path to the model.json file
     * @returns {Promise<YAMNet>}
     */
    async load(modelPath = './model/') {
        try {
            // Load the graph model
            this.model = await tf.loadGraphModel(modelPath + 'model.json');

            // Load class names
            await this.loadClassNames();

            return this;
        } catch (error) {
            throw error;
        }
    }

    /**
     * Load YAMNet class names (AudioSet classes)
     */
    async loadClassNames() {
        try {
            // Load the official YAMNet class map CSV file
            const response = await fetch('./model/assets/yamnet_class_map.csv');
            const csvText = await response.text();

            // Parse CSV and create class mapping
            this.classNames = {};
            const lines = csvText.trim().split('\n');

            // Skip header line and process each class
            for (let i = 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line) continue;

                // Parse CSV line properly, handling quoted fields with commas
                const fields = parseCSVLine(line);
                if (fields.length >= 3) {
                    const classIndex = parseInt(fields[0]);
                    const displayName = fields[2].replace(/^"|"$/g, ''); // Remove surrounding quotes
                    this.classNames[classIndex] = displayName;
                }
            }

            // Helper function to parse CSV line with proper quote handling
            function parseCSVLine(line) {
                const result = [];
                let current = '';
                let inQuotes = false;

                for (let i = 0; i < line.length; i++) {
                    const char = line[i];
                    if (char === '"') {
                        inQuotes = !inQuotes;
                    } else if (char === ',' && !inQuotes) {
                        result.push(current);
                        current = '';
                    } else {
                        current += char;
                    }
                }
                result.push(current); // Add the last field
                return result;
            }

        } catch (error) {
            // Fallback to basic mapping if CSV fails to load

            // Fallback to basic mapping if CSV fails to load
            this.classNames = {
                0: 'Speech',
                1: 'Child speech',
                4: 'Conversation',
                70: 'Bark',
                132: 'Bark',
                494: 'Silence',
                500: 'Car',
                521: 'Background noise'
            };
        }
    }

    /**
     * Make predictions on audio waveform
     * @param {Float32Array|Tensor} waveform - Audio waveform at 16kHz
     * @returns {Promise<Tensor>} Predictions tensor
     */
    async predict(waveform) {
        let inputTensor;

        // Convert to tensor if not already
        if (!(waveform instanceof tf.Tensor)) {
            inputTensor = tf.tensor(waveform);
        } else {
            inputTensor = waveform;
        }

        // For this specific YAMNet model, keep as 1D [-1] shape (no batch dimension)
        // The model expects waveform:0 to be shape [-1], not [1, -1]

        try {
            // Run inference using executeAsync with proper input name
            const predictions = await this.model.executeAsync({ 'waveform:0': inputTensor });
            return predictions[0]; // Return the first output tensor
        } finally {
            // Clean up if we created the tensor
            if (!(waveform instanceof tf.Tensor)) {
                inputTensor.dispose();
            }
        }
    }

    /**
     * Dispose of the model to free memory
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
    }
}

// Export for use in other scripts
window.YAMNet = YAMNet;

// Create a load function similar to the @tensorflow-models API
window.yamnet = {
    load: async function(modelPath = './model/') {
        const model = new YAMNet();
        await model.load(modelPath);
        return model;
    }
};
