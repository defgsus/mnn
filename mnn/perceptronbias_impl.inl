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

// ---------------- io -------------------

MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::serialize(std::ostream& s) const
{
    s << name();
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
void MNN_PERCEPTRONBIAS::brainwash()
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
    Float f = 1.0 / std::sqrt(input_.size());
    //Float f = 1.0 / input_.size();
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
    auto w = weight_.begin();
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
    for (auto o = output_.begin(); o != output_.end(); ++o, ++e)
    {
        Float de = ActFunc::derivative(*e, *o);

        Float* pd = &prevDelta_[0];
        Float* bias = &bias_[0];
        for (auto i = input_.begin(); i != input_.end(); ++i, ++w, ++pd, ++bias)
        {
            *pd = momentum_ * *pd
                + global_learn_rate * de * *i;
            *w += *pd;
            *bias += learnRateBias_ * global_learn_rate * de;
        }

    }
}


// ----------- info -----------------------

MNN_TEMPLATE
void MNN_PERCEPTRONBIAS::info(std::ostream &out) const
{
    out <<   "name       : " << name()
        << "\nlearnrate  : " << learnRate_ << " (bias: " << learnRateBias_ << ")"
        << "\nmomentum   : " << momentum_
        << "\ninputs     : " << numIn()
        << "\noutputs    : " << numOut()
        << "\nactivation : " << ActFunc::static_name()
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

