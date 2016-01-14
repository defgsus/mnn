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

    return *this;
}


// ---------------- io -------------------

MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::serialize(std::ostream& s) const
{
    s << id();
    // version
    s << " " << 1;
    // settings
    s << " " << learnRate_ << " " << momentum_;
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
    if (ver > 1)
        MNN_EXCEPTION("Wrong version in " << name());

    // settings
    s >> learnRate_ >> momentum_;
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
    bias_.resize(nrIn);
    output_.resize(nrOut);
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
            weight(nrIn * nrOut);
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
    }
    // assign new weights
    weight_ = weight;
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
    auto w = &weight_[0];
    for (auto o = output_.begin(); o != output_.end(); ++o)
    {
        Float sum = 0;
        const Float* bias = &bias_[0];
        for (auto i = input_.begin(); i != input_.end(); ++i, ++w, ++bias)
        {
            sum += (*i + *bias) * *w;
        }

        *o = ActFunc::activation(sum);
    }

    // copy to caller
    std::copy(output_.begin(), output_.end(), output);
}


MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::bprop(const Float * error, Float * error_output,
                           Float global_learn_rate)
{
    global_learn_rate *= learnRate_;

    const Float * e;

    // pass error through
    if (error_output)
    for (size_t i = 0; i<input_.size(); ++i, ++error_output)
    {
        Float sum = 0;
        e = error;
        for (size_t o = 0; o < output_.size(); ++o, ++e)
        {
            sum += *e * weight_[o * input_.size() + i];
        }
        *error_output = sum;
    }

    // backprob derivative
    e = error;
    Float* w = &weight_[0];
    Float* pd = &prevDelta_[0];
    Float* bias = &bias_[0];
    for (auto o = output_.begin(); o != output_.end(); ++o, ++e)
    {
        Float de = ActFunc::derivative(*e, *o);

        *bias += learnRateBias_ * global_learn_rate * de;

        for (auto i = input_.begin(); i != input_.end(); ++i, ++w, ++pd, ++bias)
        {
            *pd = momentum_ * *pd
                + global_learn_rate * de * *i;
            *w += *pd;
        }

    }
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
        << "\n" << pf << "activation : " << ActFunc::static_name()
        << "\n" << pf << "inputs     : " << numIn()
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

