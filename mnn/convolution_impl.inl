/** @file convolution_impl.inl

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/11/2016</p>


    input      kernel    output

    x x x x    x x       input - kernel + 1

    x x x x    x x

    x x x x


*/




#define MNN_TEMPLATE template <typename Float, class ActFunc>
#define MNN_CONVOLUTION Convolution<Float, ActFunc>

MNN_TEMPLATE
MNN_CONVOLUTION::Convolution(size_t inputWidth, size_t inputHeight, size_t inputMaps,
                             size_t kernelWidth, size_t kernelHeight, size_t outputMaps,
                             Float learnRate)
    : learnRate_	(learnRate)
    , momentum_     (.1)
{
    resize(inputWidth, inputHeight, inputMaps, kernelWidth, kernelHeight, outputMaps);
}

MNN_TEMPLATE
MNN_CONVOLUTION::Convolution(size_t inputWidth, size_t inputHeight,
                             size_t kernelWidth, size_t kernelHeight,
                             Float learnRate)
    : Convolution(inputWidth, inputHeight, 1,
                  kernelWidth, kernelHeight, 1, learnRate)
{ }

MNN_TEMPLATE
MNN_CONVOLUTION::~Convolution()
{

}

MNN_TEMPLATE
Convolution<Float, ActFunc>& MNN_CONVOLUTION::operator = (const Layer<Float>& layer)
{
    auto net = dynamic_cast<const Convolution<Float, ActFunc>*>(&layer);
    if (!net)
        return *this;

    input_ = net->input_;
    output_ = net->output_;
    weight_ = net->weight_;
    prevDelta_ = net->prevDelta_;

    inputWidth_ = net->inputWidth_;
    inputHeight_ = net->inputHeight_;
    kernelWidth_ = net->kernelWidth_;
    kernelHeight_ = net->kernelHeight_;

    learnRate_ = net->learnRate_;
    momentum_ = net->momentum_;

    return *this;
}

// ---------------- io -------------------

MNN_TEMPLATE
void MNN_CONVOLUTION::serialize(std::ostream& s) const
{
    s << id();
    // version
    s << " " << 1;
    // settings
    s << " " << learnRate_ << " " << momentum_;
    // dimension
    s << " " << inputWidth_ << " " << inputHeight_
      << " " << kernelWidth_ << " " << kernelHeight_;
    // weights
    for (auto w : weight_)
        s << " " << w;
}

MNN_TEMPLATE
void MNN_CONVOLUTION::deserialize(std::istream& s)
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
    size_t iw, ih, kw, kh;
    s >> iw >> ih >> kw >> kh;
    resize(iw, ih, kw, kh);
    // weights
    for (auto& w : weight_)
        s >> w;
}


// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_CONVOLUTION::resize(size_t inputWidth, size_t inputHeight,
                             size_t kernelWidth, size_t kernelHeight)
{
    resize(inputWidth, inputHeight, 1,
           kernelWidth, kernelHeight, 1);
}

MNN_TEMPLATE
void MNN_CONVOLUTION::resize(size_t inputWidth, size_t inputHeight, size_t inputMaps,
                             size_t kernelWidth, size_t kernelHeight, size_t outputMaps)
{
    assert(kernelWidth <= inputWidth && kernelHeight <= inputHeight
           && "input smaller than kernel size, in Convolution layer");

    inputWidth_ = inputWidth;
    inputHeight_ = inputHeight;
    kernelWidth_ = kernelWidth;
    kernelHeight_ = kernelHeight;
    inputMaps_ = inputMaps;
    outputMaps_ = outputMaps;
    scanWidth_ = inputWidth_ - kernelWidth_ + 1;
    scanHeight_ = inputHeight_ - kernelHeight_ + 1;

    input_.resize(inputWidth_ * inputHeight_ * inputMaps_);
    output_.resize(scanWidth_ * scanHeight_ * outputMaps_);
    weight_.resize(kernelWidth * kernelHeight * outputMaps_);
    prevDelta_.resize(weight_.size());
}
/*
MNN_TEMPLATE
void MNN_CONVOLUTION::grow(size_t nrIn, size_t nrOut, Float randomDev)
{
    if (nrIn < numIn() || nrOut < numOut())
        return;
}
*/

MNN_TEMPLATE
size_t MNN_CONVOLUTION::numIn() const
{
    return input_.size();
}

MNN_TEMPLATE
size_t MNN_CONVOLUTION::numOut() const
{
    return output_.size();
}


MNN_TEMPLATE
void MNN_CONVOLUTION::brainwash(Float amp)
{
    // reset in/out
    for (auto& e : input_)
        e = 0.0;
    for (auto& e : output_)
        e = 0.0;

    if (kernelWidth_ == 0 || kernelHeight_ == 0)
        return;

    // randomize weights
    Float f = amp / (kernelWidth_ * kernelHeight_);
    for (auto& w : weight_)
        w = rnd(-f, f);

    // reset momentum
    for (auto& f : prevDelta_)
        f = 0.;
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_CONVOLUTION::fprop(const Float * input, Float * output)
{
    // copy to internal data
    for (size_t i=0; i<input_.size(); ++i, ++input)
        input_[i] = *input;

    // propagate
    auto o = &output_[0];
    for (size_t om = 0; om < outputMaps_; ++om)
    for (size_t sy = 0; sy < scanHeight_; ++sy)
    for (size_t sx = 0; sx < scanWidth_; ++sx, ++o)
    {
        auto w = &weight_[om * kernelWidth_ * kernelHeight_];
        Float sum = 0;
        for (size_t iy = 0; iy < kernelHeight_; ++iy)
        for (size_t ix = 0; ix < kernelWidth_; ++ix, ++w)
        {
            sum += input_[(sy + iy) * inputWidth_ + sx + ix] * *w;
        }

        *o = ActFunc::activation(sum);
    }

    // copy to caller
    std::copy(output_.begin(), output_.end(), output);
}


MNN_TEMPLATE
void MNN_CONVOLUTION::bprop(const Float * error, Float * error_output,
                           Float global_learn_rate)
{
    global_learn_rate *= learnRate_;

    const Float * e;

    // pass error through
    // by accumulating into input field
    if (error_output)
    {
        for (size_t i=0; i<numIn(); ++i)
            error_output[i] = 0.;

        auto e = error;
        for (size_t sy = 0; sy < scanHeight_; ++sy)
        for (size_t sx = 0; sx < scanWidth_; ++sx, ++e)
        {
            auto w = &weight_[0];
            for (size_t iy = 0; iy < kernelHeight_; ++iy)
            for (size_t ix = 0; ix < kernelWidth_; ++ix, ++w)
            {
                error_output[(sy + iy) * inputWidth_ + sx + ix]
                        += *e * *w;
            }
        }
    }

    // backprob derivative
    auto o = &output_[0];
    e = error;
    for (size_t sy = 0; sy < scanHeight_; ++sy)
    for (size_t sx = 0; sx < scanWidth_; ++sx, ++o)
    {
        Float de = ActFunc::derivative(*e, *o);

        auto w = &weight_[0];
        auto pd = &prevDelta_[0];
        for (size_t iy = 0; iy < kernelHeight_; ++iy)
        for (size_t ix = 0; ix < kernelWidth_; ++ix, ++w, ++pd)
        {
            *pd = momentum_ * *pd
                + global_learn_rate * de
                    * input_[(sy + iy) * inputWidth_ + sx + ix];
            *w += *pd;
        }
    }
}


// ----------- info -----------------------

MNN_TEMPLATE
Float MNN_CONVOLUTION::getWeightAverage() const
{
    Float a = 0.;
    for (auto w : weight_)
        a += std::abs(w);
    if (!weight_.empty())
        a /= weight_.size();
    return a;
}

MNN_TEMPLATE
void MNN_CONVOLUTION::info(std::ostream& out,
                           const std::string& pf) const
{
    out <<         pf << "name       : " << name()
        << "\n" << pf << "learnrate  : " << learnRate_
        << "\n" << pf << "momentum   : " << momentum_
        << "\n" << pf << "activation : " << ActFunc::static_name()
        << "\n" << pf << "inputs     : "
                << numIn() << " (" << inputWidth_ << "x" << inputHeight_ << ")"
        << "\n" << pf << "outputs    : "
                << numOut() << " (" << scanWidth_ << "x" << scanHeight_ << ")"
        << "\n" << pf << "kernel     : " << kernelWidth_ << "x" << kernelHeight_
        << "\n" << pf << "parameters : " << numParameters()
        << "\n";
}

MNN_TEMPLATE
void MNN_CONVOLUTION::dump(std::ostream &out) const
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
#undef MNN_CONVOLUTION


