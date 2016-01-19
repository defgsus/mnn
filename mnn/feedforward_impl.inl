/** @file feedforward_impl.inl

    @brief FeedForward layer implementation

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/19/2016</p>
*/

#define MNN_TEMPLATE template <typename Float, class ActFunc>
#define MNN_FEEDFORWARD FeedForward<Float, ActFunc>

MNN_TEMPLATE
MNN_FEEDFORWARD::FeedForward(size_t nrIn, size_t nrOut, Float learnRate, bool doBias)
    : learnRate_	(learnRate)
    , learnRateBias_(.1)
    , momentum_     (.1)
    , doBias_       (doBias)
    , doSoftmax_    (false)
{
    resize(nrIn, nrOut);
}

MNN_TEMPLATE
MNN_FEEDFORWARD::~FeedForward()
{

}

MNN_TEMPLATE
FeedForward<Float, ActFunc>& MNN_FEEDFORWARD::operator = (const Layer<Float>& layer)
{
    auto net = dynamic_cast<const FeedForward<Float, ActFunc>*>(&layer);
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
    doBias_ = net->doBias_;

    return *this;
}


// ---------------- io -------------------

MNN_TEMPLATE
void MNN_FEEDFORWARD::serialize(std::ostream& s) const
{
    s << id();
    // activation
    s << " " << ActFunc::static_name();
    // version
    s << " " << 1;
    // settings
    s << " " << learnRate_ << " " << momentum_
      << " " << doBias_ << " " << doSoftmax_ << "\n";
    // dimension
    s << " " << input_.size() << " " << output_.size() << "\n";
    // bias
    if (doBias_)
        for (auto b : bias_)
            s << " " << b;
    s << "\n";
    // weights
    for (auto w : weight_)
        s << " " << w;
}

MNN_TEMPLATE
void MNN_FEEDFORWARD::deserialize(std::istream& s)
{
    std::string str;
    s >> str;
    if (str != id())
        MNN_EXCEPTION("Expected '" << id()
                      << "' in stream, found '" << str << "'");
    // activation
    s >> str;
    // version
    int ver;
    s >> ver;
    if (ver > 1)
        MNN_EXCEPTION("Wrong version in " << name());

    // settings
    s >> learnRate_ >> momentum_ >> doBias_ >> doSoftmax_;
    // dimension
    size_t numIn, numOut;
    s >> numIn >> numOut;
    resize(numIn, numOut);
    // bias
    if (doBias_)
        for (auto& b : bias_)
            s >> b;
    // weights
    for (auto& w : weight_)
        s >> w;
}


// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_FEEDFORWARD::resize(size_t nrIn, size_t nrOut)
{
    input_.resize(nrIn);
    output_.resize(nrOut);
    bias_.resize(nrOut);
    weight_.resize(nrIn * nrOut);
    prevDelta_.resize(nrIn * nrOut);
}

MNN_TEMPLATE
void MNN_FEEDFORWARD::grow(size_t nrIn, size_t nrOut, Float randomDev)
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
void MNN_FEEDFORWARD::brainwash(Float amp)
{
    // reset in/out
    for (auto& f : input_)
        f = 0.;
    for (auto& f : output_)
        f = 0.;
    // reset momentum
    for (auto& m : prevDelta_)
        m = 0.;

    if (input_.empty() || output_.empty())
        return;

    // randomize weights (assume normalized states)
    Float f = amp / std::sqrt(input_.size());
    for (auto e = weight_.begin(); e != weight_.end(); ++e)
        *e = rnd(-f, f);

    // randomize bias
    f = amp / output_.size();
    for (auto& b : bias_)
        b = rnd(-f, f);
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_FEEDFORWARD::fprop(const Float * input, Float * output)
{
    // copy to internal data
    for (auto i = input_.begin(); i != input_.end(); ++i, ++input)
        *i = *input;

    // propagate
    if (doBias_)
        DenseMatrix::fprop_bias<Float, ActFunc>(
                &input_[0], &output_[0], &bias_[0], &weight_[0],
                input_.size(), output_.size());
    else
        DenseMatrix::fprop<Float, ActFunc>(
                &input_[0], &output_[0], &weight_[0],
                input_.size(), output_.size());

    if (doSoftmax_)
        apply_softmax(&output_[0], output_.size());

    // copy to caller
    std::copy(output_.begin(), output_.end(), output);
}


MNN_TEMPLATE
void MNN_FEEDFORWARD::bprop(const Float * error, Float * error_output,
                           Float learn_rate)
{
    // calculate error derivative
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
        if (learnRate_ > 0.)
            DenseMatrix::gradient_descent<Float, Activation::Linear>(
                    &input_[0], &output_[0], &errorDer_[0],
                    &weight_[0], &prevDelta_[0],
                    input_.size(), output_.size(),
                    learn_rate * learnRate_,
                    momentum_);

        // gradient descent on biases
        if (doBias_ && learnRateBias_ > 0.)
            DenseMatrix::gradient_descent_bias<Float, Activation::Linear>(
                    &output_[0], &errorDer_[0], &bias_[0],
                    output_.size(),
                    learn_rate * learnRateBias_);
    }
}


MNN_TEMPLATE
void MNN_FEEDFORWARD::reconstruct(const Float* input, Float* reconstruction)
{
    // get code for input
    if (doBias_)
        DenseMatrix::fprop_bias<Float, ActFunc>(
                input, &output_[0], &bias_[0], &weight_[0],
                input_.size(), output_.size());
    else
        DenseMatrix::fprop<Float, ActFunc>(
                input, &output_[0], &weight_[0],
                input_.size(), output_.size());

    // get reconstruction from code
    DenseMatrix::fprop_transpose<Float, ActFunc>(
                &output_[0], reconstruction, &weight_[0],
                output_.size(), input_.size());
}

MNN_TEMPLATE
Float MNN_FEEDFORWARD::reconstructionTraining(
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
    if (doBias_)
        DenseMatrix::fprop_bias<Float, ActFunc>(
                &input_[0], &output_[0], &bias_[0], &weight_[0],
                input_.size(), output_.size());
    else
        DenseMatrix::fprop<Float, ActFunc>(
                &input_[0], &output_[0], &weight_[0],
                input_.size(), output_.size());

    if (doSoftmax_)
        apply_softmax(&output_[0], output_.size());

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

    // gradient descent on biases using code error
    if (doBias_)
        DenseMatrix::gradient_descent_bias<Float, Activation::Linear>(
                &output_[0], &reconOutput_[0], &bias_[0],
                output_.size(),
                global_learn_rate * learnRateBias_);

    return err_sum;
}



// ----------- info -----------------------

MNN_TEMPLATE
Float MNN_FEEDFORWARD::getWeightAverage() const
{
    Float a = 0.;
    for (auto w : weight_)
        a += std::abs(w);
    if (!weight_.empty())
        a /= weight_.size();
    return a;
}

MNN_TEMPLATE
void MNN_FEEDFORWARD::info(std::ostream &out, const std::string& pf) const
{
    out <<         pf << "name       : " << name()
        << "\n" << pf << "learnrate  : " << learnRate_;
    if (doBias_)
        out << " (bias: " << learnRateBias_ << ")";
    out << "\n" << pf << "momentum   : " << momentum_
        << "\n" << pf << "activation : " << ActFunc::static_name();
    if (doSoftmax_)
        out << " (softmax)";
    out << "\n" << pf << "inputs     : " << numIn()
        << "\n" << pf << "outputs    : " << numOut()
        << "\n" << pf << "parameters : " << numParameters()
        << std::endl;
}

MNN_TEMPLATE
void MNN_FEEDFORWARD::dump(std::ostream &out) const
{
    out << "inputs:";
    for (auto v : input_)
        out << " " << v;

    out << "\noutputs:";
    for (auto v : output_)
        out << " " << v;

    if (doBias_)
    {
        out << "\nbias:";
        for (auto v : bias_)
            out << " " << v;
    }

    out << "\nweights:";
    for (auto v : weight_)
        out << " " << v;

    out << std::endl;
}






#undef MNN_TEMPLATE
#undef MNN_FEEDFORWARD



