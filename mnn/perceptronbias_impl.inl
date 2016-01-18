/**	@file perceptronbias_impl.inl

    @brief PerceptronBias implementation

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/21/2015</p>
*/

#define MNN_TEMPLATE template <typename Float, class ActFunc>
#define MNN_PERCEPTRONBIAS PerceptronBias<Float, ActFunc>

MNN_TEMPLATE
MNN_PERCEPTRONBIAS::PerceptronBias(size_t nrIn, size_t nrOut, Float learnRate)
    : learnRate_	(learnRate)
    , learnRateBias_(.1)
    , momentum_     (.1)
    , doSoftmax_    (false)
{
    resize(nrIn, nrOut);
}

MNN_TEMPLATE
MNN_PERCEPTRONBIAS::~PerceptronBias()
{

}

MNN_TEMPLATE
PerceptronBias<Float, ActFunc>& MNN_PERCEPTRONBIAS::operator = (const Layer<Float>& layer)
{
    auto net = dynamic_cast<const PerceptronBias<Float, ActFunc>*>(&layer);
    if (!net)
        return *this;

    input_ = net->input_;
    bias_ = net->bias_;
    output_ = net->output_;
    weight_ = net->weight_;
    prevDelta_ = net->prevDelta_;

    learnRate_ = net->learnRate_;
    learnRateBias_ = net->learnRateBias_;
    momentum_ = net->momentum_;
    doSoftmax_ = net->doSoftmax_;

    return *this;
}


// ---------------- io -------------------

MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::serialize(std::ostream& s) const
{
    s << id();
    // version
    s << " " << 2;
    // settings
    s << " " << learnRate_ << " " << momentum_;
    // v2
    s << doSoftmax_;
    // dimension
    s << " " << numIn() << " " << numOut();
    // bias
    for (auto b : bias_)
        s << " " << b;
    // weights
    for (auto w : weight_)
        s << " " << w;
}

MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::deserialize(std::istream& s)
{
    std::string str;
    s >> str;
    if (str != id())
        MNN_EXCEPTION("Expected '" << id()
                      << "' in stream, found '" << str << "'");
    // version
    int ver;
    s >> ver;
    if (ver > 2)
        MNN_EXCEPTION("Wrong version in " << name());

    // settings
    s >> learnRate_ >> momentum_;
    if (ver > 2)
        s >> doSoftmax_;
    // dimension
    size_t numIn, numOut;
    s >> numIn >> numOut;
    resize(numIn, numOut);
    // bias
    for (auto& b : bias_)
        s >> b;
    // weights
    for (auto& w : weight_)
        s >> w;
}


// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::resize(size_t nrIn, size_t nrOut)
{
    input_.resize(nrIn);
    output_.resize(nrOut);
    bias_.resize(nrOut);
    weight_.resize(nrIn * nrOut);
    prevDelta_.resize(nrIn * nrOut);
}

MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::grow(size_t nrIn, size_t nrOut, Float randomDev)
{
    if (nrIn < numIn() || nrOut < numOut())
        return;

    // copy weights
    std::vector<Float>
            weight(nrIn * nrOut),
    // and biases
            bias(nrOut);
    size_t o;
    for (o=0; o<output_.size(); ++o)
    {
        size_t i;
        for (i=0; i<input_.size(); ++i)
            weight[o * nrIn + i] = weight_[o * input_.size() + i];
        // choose random input to copy
        size_t ri = size_t(rand()) % input_.size();
        // run through additional inputs
        for (; i<nrIn; ++i)
            weight[o * nrIn + i] = weight_[o * input_.size() + ri]
                                    + rndg(Float(0), randomDev);
        // copy biases
        bias[o] = bias_[o];
    }
    // run through additional outputs
    for (; o<nrOut; ++o)
    {
        // choose random input and output to copy
        size_t ro = size_t(rand()) % output_.size();
        size_t ri = size_t(rand()) % input_.size();

        size_t i;
        for (i=0; i<input_.size(); ++i)
            weight[o * nrIn + i] = weight_[ro * input_.size() + i];
        for (; i<nrIn; ++i)
            weight[o * nrIn + i] = weight_[ro * input_.size() + ri]
                                    + rndg(Float(0), randomDev);
        bias[o] = bias_[ro];
    }
    // assign new weights and biases
    weight_ = weight;
    bias_ = bias;
    // resize other buffers
    input_.resize(nrIn); for (auto&f : input_) f = 0;
    output_.resize(nrOut); for (auto&f : output_) f = 0;
    prevDelta_.resize(nrIn * nrOut); for (auto&f : prevDelta_) f = 0;
}

MNN_TEMPLATE
size_t MNN_PERCEPTRONBIAS::numIn() const
{
    return input_.size();
}

MNN_TEMPLATE
size_t MNN_PERCEPTRONBIAS::numOut() const
{
    return output_.size();
}


MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::brainwash(Float amp)
{
    // reset in/out
    for (auto& f : input_)
        f = 0.;
    for (auto& f : output_)
        f = 0.;
    // reset cells bias
    for (auto& f : bias_)
        f = 0.;

    if (input_.empty() || output_.empty())
        return;

    // randomize weights (assume normalized states)
    Float f = amp / std::sqrt(input_.size());
    for (auto e = weight_.begin(); e != weight_.end(); ++e)
        *e = rnd(-f, f);

    // reset momentum
    for (auto& f : prevDelta_)
        f = 0.;
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::fprop(const Float * input, Float * output)
{
    // copy to internal data
    for (auto i = input_.begin(); i != input_.end(); ++i, ++input)
        *i = *input;


    // propagate
    DenseMatrix::fprop_bias<Float, ActFunc>(
                &input_[0], &output_[0], &bias_[0], &weight_[0],
                input_.size(), output_.size());

    if (doSoftmax_)
        apply_softmax(&output_[0], output_.size());

    // copy to caller
    std::copy(output_.begin(), output_.end(), output);
}


MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::bprop(const Float * error, Float * error_output,
                           Float learn_rate)
{
    learn_rate *= learnRate_;

    // this version first calculates error derivatives
    // and passes these through to previous layers
#if 1
    // get error derivative
    if (errorDer_.size() != output_.size())
        errorDer_.resize(output_.size());
    for (size_t i=0; i<output_.size(); ++i)
        errorDer_[i] = ActFunc::derivative(error[i], output_[i]);

    // pass error through
    if (error_output)
        DenseMatrix::bprop<Float>(
                error_output, &errorDer_[0], &weight_[0],
                input_.size(), output_.size());

    if (learn_rate > 0.)
    {
        // gradient descent on weights
        DenseMatrix::gradient_descent<Float, Activation::Linear>(
                    &input_[0], &output_[0], &errorDer_[0],
                    &weight_[0], &prevDelta_[0],
                    input_.size(), output_.size(),
                    learn_rate,
                    momentum_);

        // gradient descent on biases
        if (learnRateBias_ > 0.)
        DenseMatrix::gradient_descent_bias<Float, Activation::Linear>(
                    &output_[0], &errorDer_[0], &bias_[0],
                    output_.size(),
                    learn_rate * learnRateBias_);
    }

    // this version passes errors through to previous
    // layers and uses derivatives for gradient descent
#else
    // pass error through
    if (error_output)
        DenseMatrix::bprop<Float>(
                error_output, error, &weight_[0],
                input_.size(), output_.size());

    if (learn_rate > 0.)
    {
        // gradient descent on weights
        DenseMatrix::gradient_descent<Float, ActFunc>(
                    &input_[0], &output_[0], error,
                    &weight_[0], &prevDelta_[0],
                    input_.size(), output_.size(),
                    learn_rate,
                    momentum_);

        // gradient descent on biases
        if (learnRateBias_ > 0.)
        DenseMatrix::gradient_descent_bias<Float, ActFunc>(
                    &output_[0], error, &bias_[0],
                    output_.size(),
                    learn_rate * learnRateBias_);
    }
#endif
}


MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::reconstruct(const Float* input, Float* reconstruction)
{
    // get code for input
    DenseMatrix::fprop<Float, ActFunc>(
                input, &output_[0], &weight_[0],
                input_.size(), output_.size());

    // get reconstruction from code
    DenseMatrix::fprop_transpose<Float, ActFunc>(
                &output_[0], reconstruction, &weight_[0],
                output_.size(), input_.size());
}

MNN_TEMPLATE
Float MNN_PERCEPTRONBIAS::reconstructionTraining(
            const Float *dec_input, const Float* true_input,
            Float global_learn_rate)
{
    // copy to input
    for (auto i = input_.begin(); i != input_.end(); ++i, ++dec_input)
        *i = *dec_input;

    // get reconstruction space
    if (reconInput_.size() != input_.size())
        reconInput_.resize(input_.size());
    if (reconError_.size() != input_.size())
        reconError_.resize(input_.size());
    if (reconOutput_.size() != output_.size())
        reconOutput_.resize(output_.size());

    // get code for input
    DenseMatrix::fprop_bias<Float, ActFunc>(
                &input_[0], &output_[0], &bias_[0], &weight_[0],
                input_.size(), output_.size());

    // get reconstruction from code
    DenseMatrix::fprop_transpose<Float, ActFunc>(
                &output_[0], &reconInput_[0], &weight_[0],
                output_.size(), input_.size());

    // get reconstruction error
    Float err_sum = 0.;
    auto inp = true_input,
         err = &reconError_[0];
    for (auto i : reconInput_)
    {
        Float e = *inp++ - i;
        err_sum += std::abs(e);

        *err++ = ActFunc::derivative(e, i);

    }
    err_sum /= input_.size();

    // sum errors into code vector
    DenseMatrix::fprop<Float, Activation::Linear>(
                &reconError_[0], &reconOutput_[0], &weight_[0],
                input_.size(), output_.size());

    // gradient descent using reconstruction error
    DenseMatrix::gradient_descent_transpose<Float, Activation::Linear>(
                &output_[0], &reconInput_[0], &reconError_[0],
                &weight_[0], &prevDelta_[0],
                output_.size(), input_.size(),
                global_learn_rate * learnRate_,
                momentum_);

    // gradient descent using code error
    DenseMatrix::gradient_descent<Float, Activation::Linear>(
                &input_[0], &output_[0], &reconOutput_[0],
                &weight_[0], &prevDelta_[0],
                input_.size(), output_.size(),
                global_learn_rate * learnRate_,
                momentum_);

    // gradient descent using code error
    DenseMatrix::gradient_descent_bias<Float, Activation::Linear>(
                &output_[0], &reconOutput_[0], &bias_[0],
                output_.size(),
                global_learn_rate * learnRateBias_);

    return err_sum;
}



// ----------- info -----------------------

MNN_TEMPLATE
Float MNN_PERCEPTRONBIAS::getWeightAverage() const
{
    Float a = 0.;
    for (auto w : weight_)
        a += std::abs(w);
    if (!weight_.empty())
        a /= weight_.size();
    return a;
}

MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::info(std::ostream &out, const std::string& pf) const
{
    out <<         pf << "name       : " << name()
        << "\n" << pf << "learnrate  : " << learnRate_ << " (bias: " << learnRateBias_ << ")"
        << "\n" << pf << "momentum   : " << momentum_
        << "\n" << pf << "activation : " << ActFunc::static_name();
    if (doSoftmax_)
        out << " (softmax)";
    out << "\n" << pf << "inputs     : " << numIn()
        << "\n" << pf << "outputs    : " << numOut()
        << "\n" << pf << "parameters : " << numParameters()
        << "\n";
}

MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::dump(std::ostream &out) const
{
    out << "inputs:";
    for (auto e = input_.begin(); e != input_.end(); ++e)
        out << " " << *e;

    out << "\nbias:";
    for (auto e = bias_.begin(); e != bias_.end(); ++e)
        out << " " << *e;

    out << "\noutputs:";
    for (auto e = output_.begin(); e != output_.end(); ++e)
        out << " " << *e;

    out << "\nweights:";
    for (auto e = weight_.begin(); e != weight_.end(); ++e)
        out << " " << *e;

    out << "\n";
}






#undef MNN_TEMPLATE
#undef MNN_PERCEPTRONBIAS

