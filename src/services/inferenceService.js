const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();

        const classes = ['Cancer', 'Non-cancer'];

        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;

        const classResult = tf.argMax(prediction, 1).dataSync()[0];
        const label = confidenceScore > 0.5 ? 'Cancer' : 'Non-cancer';

        let explanation, suggestion;

        if (label === 'Cancer') {
            explanation = "Cancer is detected.";
            suggestion = "Segera periksa ke dokter!";
        } else if (label === 'Non-cancer') {
            explanation = "No signs of cancer detected.";
            suggestion = "Anda sehat!";
        }

        return { confidenceScore, label, explanation, suggestion };
    } catch (error) {
        throw new InputError(`Terjadi kesalahan input: ${error.message}`);
    }
}

module.exports = predictClassification;
