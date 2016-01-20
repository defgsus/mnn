/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/12/2016</p>
*/


#define MNN_TEMPLATE template <typename Float>
#define MNN_STACKPARALLEL StackParallel<Float>

MNN_TEMPLATE
MNN_STACKPARALLEL::StackParallel()
{

}



MNN_TEMPLATE
StackParallel<Float>& MNN_STACKPARALLEL::operator = (const Layer<Float>& layer)
{
    auto net = dynamic_cast<const StackParallel<Float>*>(&layer);
    if (!net)
        return *this;

    Stack<Float>::clearLayers();

    for (size_t i = 0; i < net->numLayers(); ++i)
    {
        auto l = net->layer(i)->getCopy();
        Stack<Float>::addLayer(l);
    }

    return *this;
}


// ---------------- io -------------------

MNN_TEMPLATE
void MNN_STACKPARALLEL::serialize(std::ostream& s) const
{
    s << id();
    // version
    s << " " << 1;
    // dimension
    s << " " << Stack<Float>::numLayers() << " ";
    // each layer
    for (auto l : Stack<Float>::layer_)
    {
        s << "\n";
        l->serialize(s);
    }
}

MNN_TEMPLATE
void MNN_STACKPARALLEL::deserialize(std::istream& s)
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

    // dimension
    size_t numLay;
    s >> numLay;
    if (numLay != Stack<Float>::numLayers())
        MNN_EXCEPTION("Number of layers does not match in " << name()
                      << ", expected " << Stack<Float>::numLayers() << ", found " << numLay);
    // each layer
    for (auto l : Stack<Float>::layer_)
        l->deserialize(s);

    updateLayers();
}



// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_STACKPARALLEL::resize(size_t numIn, size_t numOut)
{
    assert(!"resize() not implemented for StackParallel");
    (void)numIn;
    (void)numOut;
}

MNN_TEMPLATE
void MNN_STACKPARALLEL::grow(size_t nrIn, size_t nrOut, Float randomDev)
{
    assert(!"grow() not implemented for StackParallel");
    (void)nrIn;
    (void)nrOut;
    (void)randomDev;
}

MNN_TEMPLATE
size_t MNN_STACKPARALLEL::numIn() const
{
    size_t num = 0;
    for (auto l : Stack<Float>::layer_)
        num += l->numIn();
    return num;
}

MNN_TEMPLATE
size_t MNN_STACKPARALLEL::numOut() const
{
    return bufferOut_.size();
}


// ----------- layer interface -----------

MNN_TEMPLATE
void MNN_STACKPARALLEL::updateLayers()
{
    {	// clear buffer (completely)
        std::vector<Float> tmp;
        tmp.swap( bufferOut_ );
    }

    if (Stack<Float>::layer_.empty())
        return;

    size_t num = 0;
    for (auto l : Stack<Float>::layer_)
        num += l->numOut();

    bufferOut_.resize(num);
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_STACKPARALLEL::fprop(const Float * input, Float * output)
{
    if (Stack<Float>::layer_.empty())
        return;

    auto b = &bufferOut_[0];
    for (auto l : Stack<Float>::layer_)
    {
        l->fprop(input, b);
        input += l->numIn();
        b += l->numOut();
    }

    // copy to caller
    for (auto i = bufferOut_.begin(); i != bufferOut_.end(); ++i, ++output)
        *output = *i;
}


MNN_TEMPLATE
void MNN_STACKPARALLEL::bprop(const Float * error, Float * error_output,
                            Float global_learn_rate)
{
    if (Stack<Float>::layer_.empty())
        return;

    for (auto l : Stack<Float>::layer_)
    {
        l->bprop(error, error_output, global_learn_rate);
        if (error_output)
            error_output += l->numIn();
        error += l->numOut();
    }
}


// ----------- info -----------------------

MNN_TEMPLATE
void MNN_STACKPARALLEL::info(std::ostream &out, const std::string& pf) const
{
    out <<         pf << "name       : " << name()
        << "\n" << pf << "layout     : ";
    if (Stack<Float>::layer_.empty())
        out << "empty";
    else
    {
        out << Stack<Float>::layer_[0]->numIn() << "-" << Stack<Float>::layer_[0]->numOut();
        for (size_t i=1; i<Stack<Float>::layer_.size(); ++i)
            out << " | " << Stack<Float>::layer_[i]->numIn()
                    << "-" << Stack<Float>::layer_[i]->numOut();
    }
    out << "\n" << pf << "parameters : " << Stack<Float>::numParameters()
        << "\n" << pf << "weights av : " << Stack<Float>::getWeightAverage()
        << "\n";
    size_t k = 1;
    for (auto l = Stack<Float>::layer_.begin(); l != Stack<Float>::layer_.end(); ++l, ++k)
    {
        out << pf << "-- parallel layer " << k << " --\n";
        (*l)->info(out, pf + "  ");
    }
}

MNN_TEMPLATE
void MNN_STACKPARALLEL::dump(std::ostream &out) const
{
    size_t k = 0;
    for (auto l = Stack<Float>::layer_.begin(); l != Stack<Float>::layer_.end(); ++l, ++k)
    {
        out << "-- parallel layer " << k << " --\n";
        (*l)->dump(out);
    }
}






#undef MNN_TEMPLATE
#undef MNN_STACKPARALLEL

