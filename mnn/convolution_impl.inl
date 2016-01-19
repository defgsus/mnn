/** @file convolution_impl.inl

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/11/2016</p>


    input       kernel    output

    x x x x x   x x       (input - kernel) / stride + 1

    x x x x x   x x

    x x x x x


*/




#define MNN_TEMPLATE template <typename Float, class ActFunc>
#define MNN_CONVOLUTION Convolution<Float, ActFunc>

MNN_TEMPLATE
MNN_CONVOLUTION::Convolution(size_t inputWidth, size_t inputHeight, size_t inputMaps,
                             size_t strideX, size_t strideY,
                             size_t kernelWidth, size_t kernelHeight, size_t outputMaps,
                             Float learnRate)
    : learnRate_	(learnRate)
    , momentum_     (.1)
{
    resize(inputWidth, inputHeight, inputMaps,
           strideX, strideY, kernelWidth, kernelHeight, outputMaps);
}

MNN_TEMPLATE
MNN_CONVOLUTION::Convolution(size_t inputWidth, size_t inputHeight, size_t inputMaps,
                             size_t kernelWidth, size_t kernelHeight, size_t outputMaps,
                             Float learnRate)
    : Convolution(inputWidth, inputHeight, inputMaps, 1, 1,
                  kernelWidth, kernelHeight, outputMaps, learnRate)
{
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
    scanWidth_ = net->scanWidth_;
    scanHeight_ = net->scanHeight_;
    strideX_ = net->strideX_;
    strideY_ = net->strideY_;
    inputMaps_ = net->inputMaps_;
    parallelMaps_ = net->parallelMaps_;

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
      << " " << kernelWidth_ << " " << kernelHeight_
      << " " << strideX_ << " " << strideY_
      << " " << inputMaps_ << " " << parallelMaps_;
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
    size_t iw, ih, kw, kh, im, pm, sx, sy;
    s >> iw >> ih >> kw >> kh >> sx >> sy >> im >> pm;
    resize(iw, ih, im, sx, sy, kw, kh, pm);
    // weights
    for (auto& w : weight_)
        s >> w;
}


// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_CONVOLUTION::resize(size_t inputWidth, size_t inputHeight, size_t inputMaps,
                             size_t strideX, size_t strideY,
                             size_t kernelWidth, size_t kernelHeight, size_t parallelMaps)
{
    assert(kernelWidth <= inputWidth && kernelHeight <= inputHeight
           && "input smaller than kernel size, in Convolution layer");

    inputWidth_ = inputWidth;
    inputHeight_ = inputHeight;
    kernelWidth_ = kernelWidth;
    kernelHeight_ = kernelHeight;
    inputMaps_ = inputMaps;
    parallelMaps_ = parallelMaps;
    strideX_ = strideX;
    strideY_ = strideY;
    scanWidth_ = (inputWidth_ - kernelWidth_) / strideX_ + 1;
    scanHeight_ = (inputHeight_ - kernelHeight_) / strideY_ + 1;

    input_.resize(inputWidth_ * inputHeight_ * inputMaps_);
    output_.resize(scanWidth_ * scanHeight_ * parallelMaps_ * inputMaps_);
    weight_.resize(kernelWidth * kernelHeight * parallelMaps_ * inputMaps_);
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

    const size_t
            mapSizeInput = inputWidth_ * inputHeight_,
            mapSizeWeight = kernelWidth_ * kernelHeight_,
            mapSizeOutput = scanWidth_ * scanHeight_;

    for (size_t om = 0; om < parallelMaps_; ++om)
    for (size_t im = 0; im < inputMaps_; ++im)
    {
        const size_t idx = (om * inputMaps_ + im);

        ConvolutionMatrix::fprop<Float, ActFunc>(
                    &input_[im * mapSizeInput],
                    &output_[idx * mapSizeOutput],
                    &weight_[idx * mapSizeWeight],
                    inputWidth_, inputHeight_,
                    kernelWidth_, kernelHeight_,
                    strideX_, strideY_);
    }

    // copy to caller
    std::copy(output_.begin(), output_.end(), output);
}


MNN_TEMPLATE
void MNN_CONVOLUTION::bprop(const Float * error, Float * error_output,
                           Float global_learn_rate)
{
    global_learn_rate *= learnRate_;

    const size_t
            mapSizeInput = inputWidth_ * inputHeight_,
            mapSizeWeight = kernelWidth_ * kernelHeight_,
            mapSizeOutput = scanWidth_ * scanHeight_;

    // get error derivatives
    if (outputErr_.size() != output_.size())
        outputErr_.resize(output_.size());
    for (size_t i=0; i<output_.size(); ++i)
        outputErr_[i] = ActFunc::derivative(error[i], output_[i]);

    // pass error through
    // by accumulating into input fields
    if (error_output)
    {
        for (size_t om = 0; om < parallelMaps_; ++om)
        for (size_t im = 0; im < inputMaps_; ++im)
        {
            const size_t idx = (om * inputMaps_ + im);

            ConvolutionMatrix::bprop<Float>(
                        &error_output   [im * mapSizeInput],
                        &outputErr_     [idx * mapSizeOutput],
                        &weight_        [idx * mapSizeWeight],
                        inputWidth_, inputHeight_,
                        kernelWidth_, kernelHeight_,
                        strideX_, strideY_);
        }
    }

    // adjust weights
    if (weightBuffer_.size() != mapSizeWeight)
        weightBuffer_.resize(mapSizeWeight);
    for (size_t om = 0; om < parallelMaps_; ++om)
    for (size_t im = 0; im < inputMaps_; ++im)
    {
        const size_t idx = (om * inputMaps_ + im);

        ConvolutionMatrix::gradient_descent<Float>(
                    &input_     [im * mapSizeInput],
                    &outputErr_ [idx * mapSizeOutput],
                    &weight_    [idx * mapSizeWeight],
                    &prevDelta_ [idx * mapSizeWeight],
                    &weightBuffer_[0],
                    inputWidth_, inputHeight_,
                    kernelWidth_, kernelHeight_,
                    strideX_, strideY_,
                    global_learn_rate, momentum_);
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
        << "\n" << pf << "inputs     : " << numIn() << " ("
                      << inputWidth_ << "x" << inputHeight_;
    if (inputMaps_ > 1)
        out << " x " << inputMaps_;
    out << ")";
    if (strideX_ > 1 || strideY_ > 1)
        out << "\n" << pf << "stride     : " << strideX_ << "x" << strideY_;
    out << "\n" << pf << "outputs    : " << numOut() << " ("
                      << scanWidth_ << "x" << scanHeight_;
    const size_t outMaps = numOutputMaps();
    if (outMaps > 1)
        out << " x " << outMaps;
    out << ")";
    out << "\n" << pf << "kernel     : " << kernelWidth_ << "x" << kernelHeight_;
    if (outMaps > 1)
        out << " x " << outMaps;
    out << "\n" << pf << "parameters : " << numParameters()
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


