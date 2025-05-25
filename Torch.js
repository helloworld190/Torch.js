class Tensor {
    constructor(data, shape) {
        if (Array.isArray(data) && shape === undefined) {
            this.data = data;
            if (data.length > 0 && Array.isArray(data[0])) {
                this.shape = [data.length, data[0].length];
            } else if (data.length > 0) {
                this.shape = [1, data.length];
                this.data = [data];
            } else {
                this.shape = [0, 0];
            }
        } else if (typeof data === 'number' && Array.isArray(shape)) {
            this.shape = shape;
            this.data = Tensor._createData(shape, data);
        } else if (data === undefined && Array.isArray(shape)) {
            this.shape = shape;
            this.data = Tensor._createData(shape, 0);
        } else {
            this.data = [];
            this.shape = [0,0];
        }
    }

    static _createData(shape, fillValue) {
        if (shape.length === 1) return Array(shape[0]).fill(fillValue);
        if (shape.length === 2) {
            return Array(shape[0]).fill(null).map(() => Array(shape[1]).fill(fillValue));
        }
        throw new Error('Tensor supports 1D or 2D shapes for creation with fill value.');
    }

    static fromArray(array) {
        return new Tensor(array);
    }

    static zeros(rows, cols) {
        const shape = cols === undefined ? [rows] : [rows, cols];
        return new Tensor(0, shape);
    }

    static ones(rows, cols) {
        const shape = cols === undefined ? [rows] : [rows, cols];
        return new Tensor(1, shape);
    }

    static random(rows, cols, min = -1, max = 1) {
        const shape = cols === undefined ? [rows] : [rows, cols];
        const data = Tensor._createData(shape, 0);
        const range = max - min;
        for (let i = 0; i < shape[0]; i++) {
            if (shape.length === 2) {
                for (let j = 0; j < shape[1]; j++) {
                    data[i][j] = Math.random() * range + min;
                }
            } else {
                 data[i] = Math.random() * range + min;
            }
        }
        return new Tensor(data, shape);
    }

    get rows() { return this.shape[0]; }
    get cols() { return this.shape.length > 1 ? this.shape[1] : 1; }


    map(func) {
        const newData = this.data.map(row => row.map(val => func(val)));
        return new Tensor(newData);
    }

    T() {
        if (this.shape.length === 1) return new Tensor([this.data], [this.shape[0], 1]);
        const newData = Array(this.cols).fill(null).map(() => Array(this.rows).fill(0));
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                newData[j][i] = this.data[i][j];
            }
        }
        return new Tensor(newData);
    }

    dot(other) {
        if (this.cols !== other.rows) throw new Error(`Dimension mismatch for dot product: ${this.shape} x ${other.shape}`);
        const resultData = Array(this.rows).fill(null).map(() => Array(other.cols).fill(0));
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < other.cols; j++) {
                let sum = 0;
                for (let k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                resultData[i][j] = sum;
            }
        }
        return new Tensor(resultData);
    }

    _elementWiseOp(other, op) {
        if (typeof other === 'number') {
            return this.map(val => op(val, other));
        }
        if (this.rows !== other.rows || this.cols !== other.cols) {
            if (other.rows === 1 && other.cols === this.cols) { // Broadcast row vector
                const newData = this.data.map(row => row.map((val, j) => op(val, other.data[0][j])));
                return new Tensor(newData);
            }
            throw new Error(`Dimension mismatch for element-wise operation: ${this.shape} vs ${other.shape}`);
        }
        const newData = this.data.map((row, i) => row.map((val, j) => op(val, other.data[i][j])));
        return new Tensor(newData);
    }

    add(other) {
        return this._elementWiseOp(other, (a, b) => a + b);
    }

    subtract(other) {
        return this._elementWiseOp(other, (a, b) => a - b);
    }

    multiply(other) {
        return this._elementWiseOp(other, (a, b) => a * b);
    }

    divide(other) {
        return this._elementWiseOp(other, (a, b) => a / b);
    }
    
    pow(exponent) {
        return this.map(val => Math.pow(val, exponent));
    }

    sum(axis = -1) {
        if (axis === -1) { // sum all elements
            return this.data.reduce((acc, row) => acc + row.reduce((s, val) => s + val, 0), 0);
        }
        if (axis === 0) { // sum along columns (result is a row vector)
            const sums = Array(this.cols).fill(0);
            for (let j = 0; j < this.cols; j++) {
                for (let i = 0; i < this.rows; i++) {
                    sums[j] += this.data[i][j];
                }
            }
            return new Tensor([sums]);
        }
        if (axis === 1) { // sum along rows (result is a column vector)
            const sums = this.data.map(row => row.reduce((s, val) => s + val, 0));
            return new Tensor(sums.map(s => [s]));
        }
        throw new Error('Invalid axis for sum');
    }
    
    mean(axis = -1) {
        if (axis === -1) {
            return this.sum() / (this.rows * this.cols);
        }
        const summed = this.sum(axis);
        const count = (axis === 0) ? this.rows : this.cols;
        return summed.map(v => v / count);
    }

    clone() {
        return new Tensor(JSON.parse(JSON.stringify(this.data)));
    }

    print() {
        console.table(this.data);
    }
}

const Activations = {
    linear: {
        forward: x => x,
        backward: () => 1 
    },
    sigmoid: {
        forward: x => 1 / (1 + Math.exp(-x)),
        backward: y => y * (1 - y) 
    },
    relu: {
        forward: x => Math.max(0, x),
        backward: y => y > 0 ? 1 : 0 
    },
    tanh: {
        forward: x => Math.tanh(x),
        backward: y => 1 - y * y 
    },
    softmax: {
        forward: (inputTensor) => {
            const maxVal = Math.max(...inputTensor.data[0]);
            const exps = inputTensor.map(x => Math.exp(x - maxVal));
            const sumExps = exps.sum();
            return exps.map(x => x / sumExps);
        },
        backward: (outputTensor) => {
            return outputTensor.map(y => y * (1 - y));
        }
    }
};

const Losses = {
    mse: {
        forward: (yPred, yTrue) => {
            if (yPred.shape[0] !== yTrue.shape[0] || yPred.shape[1] !== yTrue.shape[1]) {
                 throw new Error(`MSE Loss: Shape mismatch ${yPred.shape} vs ${yTrue.shape}`);
            }
            return yPred.subtract(yTrue).pow(2).mean();
        },
        backward: (yPred, yTrue) => {
            if (yPred.shape[0] !== yTrue.shape[0] || yPred.shape[1] !== yTrue.shape[1]) {
                 throw new Error(`MSE Gradient: Shape mismatch ${yPred.shape} vs ${yTrue.shape}`);
            }
            return yPred.subtract(yTrue).multiply(2 / (yPred.cols * yPred.rows));
        }
    },
    crossEntropy: {
        forward: (yPred, yTrue) => {
            if (yPred.shape[0] !== yTrue.shape[0] || yPred.shape[1] !== yTrue.shape[1]) {
                 throw new Error(`CrossEntropy Loss: Shape mismatch ${yPred.shape} vs ${yTrue.shape}`);
            }
            const N = yPred.rows;
            const M = yPred.cols;
            let loss = 0;
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < M; j++) {
                    loss -= yTrue.data[i][j] * Math.log(yPred.data[i][j] + 1e-9); 
                }
            }
            return loss / N;
        },
        backward: (yPred, yTrue) => {
            if (yPred.shape[0] !== yTrue.shape[0] || yPred.shape[1] !== yTrue.shape[1]) {
                 throw new Error(`CrossEntropy Gradient: Shape mismatch ${yPred.shape} vs ${yTrue.shape}`);
            }
            const N = yPred.rows;
            return yTrue.divide(yPred.map(x => x + 1e-9)).multiply(-1/N);
        }
    },
    softmaxCrossEntropy: {
        forward: (preSoftmaxOutput, yTrue) => {
            const N = preSoftmaxOutput.rows;
            const M = preSoftmaxOutput.cols;
            let totalLoss = 0;
            for(let i=0; i<N; i++) {
                const row = preSoftmaxOutput.data[i];
                const maxVal = Math.max(...row);
                const exps = row.map(x => Math.exp(x - maxVal));
                const sumExps = exps.reduce((a,b) => a + b, 0);
                for(let j=0; j<M; j++) {
                    const p_ij = exps[j] / sumExps;
                    totalLoss -= yTrue.data[i][j] * Math.log(p_ij + 1e-9);
                }
            }
            return totalLoss / N;
        },
        backward: (preSoftmaxOutput, yTrue) => {
            const N = preSoftmaxOutput.rows;
            const M = preSoftmaxOutput.cols;
            const gradientData = Tensor._createData([N, M], 0);

            for(let i=0; i<N; i++) {
                const row = preSoftmaxOutput.data[i];
                const maxVal = Math.max(...row);
                const exps = row.map(x => Math.exp(x - maxVal));
                const sumExps = exps.reduce((a,b) => a + b, 0);
                const softmaxOutputs = exps.map(x => x / sumExps);
                
                for(let j=0; j<M; j++) {
                    gradientData[i][j] = (softmaxOutputs[j] - yTrue.data[i][j]) / N;
                }
            }
            return new Tensor(gradientData);
        }
    }
};


class Layer {
    constructor() {
        this.input = null;
        this.output = null;
    }
    forward(inputTensor, isTraining) { throw new Error('Not implemented'); }
    backward(outputGradient) { throw new Error('Not implemented'); }
    updateParameters(optimizer, learningRate) { }
}

class DenseLayer extends Layer {
    constructor({ inputSize, outputSize }) {
        super();
        this.weights = Tensor.random(inputSize, outputSize, -1/Math.sqrt(inputSize), 1/Math.sqrt(inputSize));
        this.biases = Tensor.zeros(1, outputSize);
        this.dWeights = null;
        this.dBiases = null;
    }

    forward(inputTensor, isTraining = false) {
        this.input = inputTensor;
        this.output = this.input.dot(this.weights).add(this.biases);
        return this.output;
    }

    backward(outputGradient) {
        this.dWeights = this.input.T().dot(outputGradient);
        this.dBiases = outputGradient.sum(0); 
        const inputGradient = outputGradient.dot(this.weights.T());
        return inputGradient;
    }

    updateParameters(optimizer, learningRate) {
        optimizer.update(this.weights, this.dWeights, learningRate);
        optimizer.update(this.biases, this.dBiases, learningRate);
    }
}

class ActivationLayer extends Layer {
    constructor(activationName) {
        super();
        if (!Activations[activationName]) throw new Error(`Activation ${activationName} not found.`);
        this.activation = Activations[activationName];
        this.isSoftmax = activationName === 'softmax';
    }

    forward(inputTensor, isTraining = false) {
        this.input = inputTensor;
        if (this.isSoftmax) {
            this.output = this.activation.forward(this.input);
        } else {
            this.output = this.input.map(this.activation.forward);
        }
        return this.output;
    }

    backward(outputGradient) {
        if (this.isSoftmax) {
             return outputGradient; // Special handling with softmaxCrossEntropy often means gradient passes through as is or is combined
        }
        const activationDerivative = this.output.map(this.activation.backward);
        return outputGradient.multiply(activationDerivative);
    }
}


class Optimizer {
    constructor({ learningRate = 0.01 } = {}) {
        this.learningRate = learningRate;
    }
    update(param, grad, learningRateOverride) { throw new Error('Not implemented'); }
}

class SGD extends Optimizer {
    constructor({ learningRate = 0.01 } = {}) {
        super({ learningRate });
    }

    update(param, grad, learningRateOverride) {
        const lr = learningRateOverride || this.learningRate;
        if (!param || !grad) return;
        if (param.shape[0] !== grad.shape[0] || param.shape[1] !== grad.shape[1]) {
             if (param.shape[0] === 1 && param.shape[1] === grad.shape[1] && grad.shape[0] > 1) { // Bias case
                 // grad is likely summed, shape will match
             } else {
                throw new Error(`SGD update: Param/Grad shape mismatch. Param: ${param.shape}, Grad: ${grad.shape}`);
             }
        }
        param.data = param.subtract(grad.multiply(lr)).data;
    }
}

class Adam extends Optimizer {
    constructor({ learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 } = {}) {
        super({ learningRate });
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.m = new Map();
        this.v = new Map();
        this.t = 0;
    }

    update(param, grad, learningRateOverride) {
        const lr = learningRateOverride || this.learningRate;
        if (!param || !grad) return;
        if (!this.m.has(param)) {
            this.m.set(param, Tensor.zeros(param.rows, param.cols));
            this.v.set(param, Tensor.zeros(param.rows, param.cols));
        }

        this.t++;
        
        let mt = this.m.get(param);
        let vt = this.v.get(param);

        mt.data = mt.multiply(this.beta1).add(grad.multiply(1 - this.beta1)).data;
        vt.data = vt.multiply(this.beta2).add(grad.pow(2).multiply(1 - this.beta2)).data;
        
        const mt_hat = mt.divide(1 - Math.pow(this.beta1, this.t));
        const vt_hat = vt.divide(1 - Math.pow(this.beta2, this.t));
        
        const updateVec = mt_hat.divide(vt_hat.map(Math.sqrt).add(this.epsilon));
        param.data = param.subtract(updateVec.multiply(lr)).data;
    }
}


class SequentialModel {
    constructor() {
        this.layers = [];
        this.optimizer = null;
        this.lossFunction = null;
        this.lossName = '';
    }

    addLayer(layer) {
        this.layers.push(layer);
    }

    compile({ optimizer, loss }) {
        this.optimizer = optimizer || new SGD();
        if (typeof loss === 'string') {
            if (!Losses[loss]) throw new Error(`Loss function ${loss} not found.`);
            this.lossFunction = Losses[loss];
            this.lossName = loss;
        } else {
            this.lossFunction = loss; 
            this.lossName = 'custom';
        }
    }

    predict(inputTensor, { thinkingTime = 0 } = {}) {
        let output = inputTensor;
        for (const layer of this.layers) {
            output = layer.forward(output, false); 
        }
        return output;
    }

    async train(X_train, y_train, { epochs = 10, batchSize = 1, learningRate, verbose = true, thinkingTimePerBatch = 0 } = {}) {
        if (!this.optimizer || !this.lossFunction) throw new Error('Model must be compiled before training.');
        
        const numSamples = X_train.rows;
        const lr = learningRate || (this.optimizer ? this.optimizer.learningRate : 0.01);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let numBatchesProcessed = 0;

            const indices = Array.from({length: numSamples}, (_, i) => i);
            for (let i = 0; i < numSamples; i += batchSize) {
                const batchIndices = indices.slice(i, Math.min(i + batchSize, numSamples));
                if (batchIndices.length === 0) continue;

                const x_batch_data = batchIndices.map(idx => X_train.data[idx]);
                const y_batch_data = batchIndices.map(idx => y_train.data[idx]);
                
                const x_batch = new Tensor(x_batch_data);
                const y_batch = new Tensor(y_batch_data);

                let output = x_batch;
                for (const layer of this.layers) {
                    output = layer.forward(output, true);
                }
                
                let loss;
                let grad;

                if (this.lossName === 'softmaxCrossEntropy' && this.layers[this.layers.length-1] instanceof ActivationLayer && this.layers[this.layers.length-1].isSoftmax) {
                    const preSoftmaxOutput = this.layers[this.layers.length-1].input;
                    loss = this.lossFunction.forward(preSoftmaxOutput, y_batch);
                    grad = this.lossFunction.backward(preSoftmaxOutput, y_batch);
                } else {
                    loss = this.lossFunction.forward(output, y_batch);
                    grad = this.lossFunction.backward(output, y_batch);
                }
                epochLoss += loss * batchIndices.length;
                numBatchesProcessed += batchIndices.length;

                for (let j = this.layers.length - 1; j >= 0; j--) {
                    grad = this.layers[j].backward(grad);
                }

                for (const layer of this.layers) {
                    if (layer.updateParameters) {
                        layer.updateParameters(this.optimizer, lr);
                    }
                }
                 if (thinkingTimePerBatch > 0) await new Promise(resolve => setTimeout(resolve, thinkingTimePerBatch));
            }
            if (verbose) {
                console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${(epochLoss / numBatchesProcessed).toFixed(4)}`);
            }
        }
    }
}

class Agent {
    constructor({ model, actionSpaceSize, epsilon = 0.1, epsilonDecay = 0.995, minEpsilon = 0.01, gamma = 0.99, learningStrategy = 'q_learning' }) {
        this.model = model; 
        this.actionSpaceSize = actionSpaceSize;
        this.epsilon = epsilon;
        this.epsilonDecay = epsilonDecay;
        this.minEpsilon = minEpsilon;
        this.gamma = gamma;
        this.learningStrategy = learningStrategy; 
        this.memory = []; 
        this.memoryMaxSize = 10000;
    }

    getAction(stateTensor, { thinkingTime = 0 } = {}) {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.actionSpaceSize);
        } else {
            const q_values = this.model.predict(stateTensor);
            let maxQ = -Infinity;
            let bestAction = 0;
            for(let i = 0; i < q_values.cols; i++) {
                if(q_values.data[0][i] > maxQ) {
                    maxQ = q_values.data[0][i];
                    bestAction = i;
                }
            }
            return bestAction;
        }
    }

    remember(state, action, reward, nextState, done) {
        this.memory.push({ state, action, reward, nextState, done });
        if (this.memory.length > this.memoryMaxSize) {
            this.memory.shift();
        }
    }

    async replay(batchSize, { thinkingTimePerBatch = 0 } = {}) {
        if (this.memory.length < batchSize || !this.model.optimizer) return;

        const miniBatchIndices = [];
        for (let i = 0; i < batchSize; i++) {
            miniBatchIndices.push(Math.floor(Math.random() * this.memory.length));
        }
        
        const states = [], nextStates = [];
        miniBatchIndices.forEach(idx => {
            states.push(this.memory[idx].state.data[0]);
            if (this.memory[idx].nextState) nextStates.push(this.memory[idx].nextState.data[0]);
        });

        const statesTensor = new Tensor(states);
        const q_current_batch = this.model.predict(statesTensor);
        let q_next_batch;
        if (nextStates.length > 0) {
            const nextStatesTensor = new Tensor(nextStates);
            q_next_batch = this.model.predict(nextStatesTensor);
        }


        const X_data = [];
        const y_data = [];

        for (let i = 0; i < miniBatchIndices.length; i++) {
            const idx = miniBatchIndices[i];
            const { state, action, reward, nextState, done } = this.memory[idx];
            
            let target_q_values_for_state = q_current_batch.data[i].slice();

            if (done) {
                target_q_values_for_state[action] = reward;
            } else {
                let max_next_q = -Infinity;
                if (q_next_batch && q_next_batch.data[i]) {
                    for(let j=0; j< q_next_batch.data[i].length; j++){
                        if(q_next_batch.data[i][j] > max_next_q) max_next_q = q_next_batch.data[i][j];
                    }
                } else {
                    max_next_q = 0; // Should not happen if nextState is not null
                }
                target_q_values_for_state[action] = reward + this.gamma * max_next_q;
            }
            X_data.push(state.data[0]);
            y_data.push(target_q_values_for_state);
        }

        if (X_data.length > 0) {
            const X_train = new Tensor(X_data);
            const y_train = new Tensor(y_data);
            await this.model.train(X_train, y_train, { epochs: 1, batchSize: X_train.rows, verbose: false, thinkingTimePerBatch });
        }

        if (this.epsilon > this.minEpsilon) {
            this.epsilon *= this.epsilonDecay;
        }
    }
}

class RLTrainer {
    constructor({ agent, environment }) {
        this.agent = agent;
        this.environment = environment;
    }

    async run({ episodes = 100, maxStepsPerEpisode = 200, replayBatchSize = 32, thinkingTimePerStep = 0, thinkingTimePerReplay = 0 } = {}) {
        for (let e = 0; e < episodes; e++) {
            let state = this.environment.reset();
            let totalReward = 0;
            for (let step = 0; step < maxStepsPerEpisode; step++) {
                const action = this.agent.getAction(state, { thinkingTime: thinkingTimePerStep });
                const { nextState, reward, done } = this.environment.step(action);
                this.agent.remember(state, action, reward, nextState, done);
                state = nextState;
                totalReward += reward;

                await this.agent.replay(replayBatchSize, { thinkingTimePerBatch: thinkingTimePerReplay });

                if (done) break;
            }
            console.log(`Episode: ${e + 1}/${episodes}, Total Reward: ${totalReward}, Epsilon: ${this.agent.epsilon.toFixed(3)}`);
        }
    }
}

class Task {
    constructor({ id, model, preprocessor, postprocessor, inputConfig = {}, outputConfig = {} }) {
        this.id = id;
        this.model = model;
        this.preprocessor = preprocessor || (data => data);
        this.postprocessor = postprocessor || (data => data);
        this.inputConfig = inputConfig;
        this.outputConfig = outputConfig;
    }

    async process(rawData, { thinkingTime = 0 } = {}) {
        let inputData = this.preprocessor(rawData, this.inputConfig);
        if (!(inputData instanceof Tensor)) {
            inputData = Tensor.fromArray(inputData.data || inputData);
        }
        
        const modelOutput = this.model.predict(inputData, { thinkingTime });
        
        let finalOutput = this.postprocessor(modelOutput, this.outputConfig);
        return finalOutput;
    }

    async trainModel(X_train_raw, y_train_raw, trainOptions) {
        let X_train = this.preprocessor(X_train_raw, this.inputConfig);
        let y_train = this.preprocessor(y_train_raw, this.inputConfig); // Or a specific target preprocessor

        if (!(X_train instanceof Tensor)) X_train = Tensor.fromArray(X_train.data || X_train);
        if (!(y_train instanceof Tensor)) y_train = Tensor.fromArray(y_train.data || y_train);
        
        await this.model.train(X_train, y_train, trainOptions);
    }
}


export default {
    Tensor,
    Activations,
    Losses,
    Layer,
    DenseLayer,
    ActivationLayer,
    Optimizer,
    SGD,
    Adam,
    SequentialModel,
    Agent,
    RLTrainer,
    Task
};
