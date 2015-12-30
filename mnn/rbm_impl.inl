/** @file rbm_impl.inl

    @brief Restricted Boltzman Machine implementation

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/22/2015</p>
*/

#define MNN_TEMPLATE template <typename Float, class ActFunc>
#define MNN_RBM Rbm<Float, ActFunc>

MNN_TEMPLATE
MNN_RBM::Rbm(size_t nrIn, size_t nrOut, Float learnRate, bool bc)
    : learnRate_	(learnRate)
    , momentum_     (.1)
    , biasCell_     (bc)
{
    resize(nrIn, nrOut);
}

MNN_TEMPLATE
MNN_RBM::~Rbm()
{

}

// ---------------- io -------------------

MNN_TEMPLATE
void MNN_RBM::serialize(std::ostream& s) const
{
    s << name();
    // version
    s << " " << 1;
    // settings
    s << " " << learnRate_ << " " << momentum_ << " " << biasCell_;
    // dimension
    s << " " << numIn() << " " << numOut();
    // weights
    for (auto w : weight_)
        s << " " << w;
}

MNN_TEMPLATE
void MNN_RBM::deserialize(std::istream& s)
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
    s >> learnRate_ >> momentum_ >> biasCell_;
    // dimension
    size_t numIn, numOut;
    s >> numIn >> numOut;
    resize(numIn, numOut);
    // weights
    for (auto& w : weight_)
        s >> w;
}

// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_RBM::resize(size_t nrIn, size_t nrOut)
{
    if (biasCell_)
        ++nrIn;
    input_.resize(nrIn);
    output_.resize(nrOut);
    weight_.resize(nrIn * nrOut);
    prevDelta_.resize(nrIn * nrOut);
    correlationData_.resize(nrIn * nrOut);
    correlationModel_.resize(nrIn * nrOut);
}


MNN_TEMPLATE
void MNN_RBM::grow(size_t nrIn, size_t nrOut, Float randomDev)
{
    if (nrIn < numIn() || nrOut < numOut())
        return;

    if (biasCell_)
        ++nrIn;

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
    if (biasCell_) input_[input_.size()-1] = 1;
    output_.resize(nrOut); for (auto&f : output_) f = 0;
    prevDelta_.resize(nrIn * nrOut); for (auto&f : prevDelta_) f = 0;
    correlationData_.resize(nrIn * nrOut); for (auto&f : correlationData_) f = 0;
    correlationModel_.resize(nrIn * nrOut); for (auto&f : correlationData_) f = 0;
}

MNN_TEMPLATE
size_t MNN_RBM::numIn() const
{
    return biasCell_? input_.size() - 1 : input_.size();
}

MNN_TEMPLATE
size_t MNN_RBM::numOut() const
{
    return output_.size();
}


MNN_TEMPLATE
void MNN_RBM::brainwash()
{
    // reset in/out
    for (auto e = input_.begin(); e != input_.end(); ++e)
        *e = 0.0;
    for (auto e = output_.begin(); e != output_.end(); ++e)
        *e = 0.0;

    if (input_.empty() || output_.empty())
        return;

    if (biasCell_)
        input_.back() = 1.;

    // randomize weights (assume normalized states)
    Float f = 1.0 / std::sqrt(input_.size());
    //Float f = 1.0 / input_.size();
    for (auto& w : weight_)
        w = rnd(-f, f);

    // reset momentum
    for (auto& f : prevDelta_)
        f = 0.;
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_RBM::fprop(const Float * input, Float * output)
{
    copyInput_(input);

    propUp_();

    // copy to caller
    std::copy(output_.begin(), output_.end(), output);
}


MNN_TEMPLATE
void MNN_RBM::bprop(const Float * error, Float * error_output,
                           Float global_learn_rate)
{
    global_learn_rate *= learnRate_;

    const Float * e;

    // pass error through
    if (error_output)
    for (size_t i = 0; i<numIn(); ++i, ++error_output)
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
    Float* w = &weight_[0];
    for (auto o = output_.begin(); o != output_.end(); ++o, ++error)
    {
        Float de = ActFunc::derivative(*error, *o);

        Float* pd = &prevDelta_[0];
        for (auto i = input_.begin(); i != input_.end(); ++i, ++w, ++pd)
        {
            *pd = momentum_ * *pd
                + global_learn_rate * de * *i;
            *w += *pd;
        }
    }

}

MNN_TEMPLATE
Float MNN_RBM::cd(const Float* input, size_t numSteps, Float learn_rate)
{
    copyInput_(input);

    // -- CD1 --
    if (numSteps <= 1)
    {
        propUp_();
        makeBinaryOutput_();
        getCorrelation_(&correlationData_[0]);
        propDown_();
        getCorrelation_(&correlationModel_[0]);
        // train weights with correlation error
        return trainCorrelation_(learn_rate);
    }

    // -- CD with n > 1 --

    for (size_t i = 1; i < numSteps; ++i)
    {
        propUp_();
        makeBinaryOutput_();
        if (i == 1)
            getCorrelation_(&correlationData_[0]);

        propDown_();
        makeBinaryInput_();
    }

    // last CD step uses probabilities instead of binary states
    propUp_();
    propDown_();
    getCorrelation_(&correlationModel_[0]);

    // train weights with correlation error
    return trainCorrelation_(learn_rate);
}

MNN_TEMPLATE
Float MNN_RBM::compareInput(const Float* input) const
{
    Float d = 0.;
    const size_t num = numIn();
    for (size_t i=0; i<num; ++i, ++input)
        d += std::abs(*input - input_[i]);
    return d;
}

MNN_TEMPLATE
Float MNN_RBM::trainCorrelation_(Float learn_rate)
{
    learn_rate *= learnRate_;
    Float err_sum = 0.,
          *w = &weight_[0],
          *pd = &prevDelta_[0],
          *cd = &correlationData_[0],
          *cm = &correlationModel_[0];
    for (size_t j = 0; j < output_.size(); ++j)
    for (size_t i = 0; i < input_.size(); ++i, ++w, ++pd, ++cd, ++cm)
    {
        const Float de = *cd - *cm;
        err_sum += std::abs(de);

        if (learn_rate > 0.)
        {
            *pd = momentum_ * *pd
                + learn_rate * de;
            *w += *pd;

            //if (std::abs(*w) > 5.)
            //    *w *= 0.9999;
        }
    }

    return err_sum;
}


MNN_TEMPLATE
void MNN_RBM::copyInput_(const Float* input)
{
    // copy to internal data
    if (!biasCell_)
        for (auto i = input_.begin(); i != input_.end(); ++i, ++input)
            *i = *input;
    else
        for (size_t i=0; i<input_.size(); ++i, ++input)
            input_[i] = *input;
}


MNN_TEMPLATE
void MNN_RBM::getCorrelation_(Float* matrix) const
{
    for (auto j : output_)
        for (auto i : input_)
            *matrix++ = i * j;
}





MNN_TEMPLATE
void MNN_RBM::propUp_()
{
    Float* w = &weight_[0];

    for (auto o = output_.begin(); o != output_.end(); ++o)
    {
        Float sum = 0;
        for (auto i = input_.begin(); i != input_.end(); ++i, ++w)
        {
            sum += *i * *w;
        }

        *o = ActFunc::activation(sum);
    }
}

MNN_TEMPLATE
void MNN_RBM::propDown_()
{
    for (size_t i = 0; i < input_.size(); ++i)
    {
        Float sum = 0;
        for (size_t j = 0; j < output_.size(); ++j)
        {
            sum += output_[j] * weight_[j * input_.size() + i];
        }

        input_[i] = ActFunc::activation(sum);
    }
}

MNN_TEMPLATE
void MNN_RBM::makeBinary_(Float* states, size_t num)
{
    for (size_t i=0; i<num; ++i, ++states)
    {
        *states = *states > rnd(Float(0), Float(1))
                ? Float(1) : Float(0);
    }
}


// ----------- info -----------------------

MNN_TEMPLATE
Float MNN_RBM::getWeightAverage() const
{
    Float a = 0.;
    for (auto w : weight_)
        a += std::abs(w);
    if (!weight_.empty())
        a /= weight_.size();
    return a;
}


MNN_TEMPLATE
void MNN_RBM::info(std::ostream &out) const
{
    out <<   "name       : " << name()
        << "\nlearnrate  : " << learnRate_
        << "\nmomentum   : " << momentum_
        << "\ninputs     : " << numIn()
            << (biasCell_ ? " (+1 bias)" : "")
        << "\noutputs    : " << numOut()
        << "\nactivation : " << ActFunc::static_name()
        << "\n";
}

MNN_TEMPLATE
void MNN_RBM::dump(std::ostream &out) const
{
    out << "inputs:";
    for (auto e = input_.begin(); e != input_.end(); ++e)
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
#undef MNN_RBM

