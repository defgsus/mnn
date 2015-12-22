#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "mnn/mnn.h"
//#include "trainposition.h"
#include "trainmnist.h"
#include "mnistset.h"


/** Brainwash and train a network with @p nrSamples of
    @p nrIn floats each and @p nrSamples expected results.
    Returns average error over each input sample
    and number of epochs */
template <typename Float, class Net>
std::pair<Float, size_t>
      simple_test(Net& net, size_t nrIn, size_t nrSamples,
                 std::vector<Float>& input, std::vector<Float>& result,
                 bool doPrint = true)
{
	Float output;

	net.resize(nrIn, 1);
	net.brainwash();
    if (doPrint)
    {
        net.info();
        net.dump();
        std::cout << "training simple pattern with "<<nrIn<<" inputs...\n";
    }

	// stochastic gradient descent
	Float error = 1.0, e;
	size_t count = 0;

	while (error > 0.01 && count<100000)
	{
		// choose pattern
		size_t nr = rand()%nrSamples;
		// feed forward
		net.fprop(&input[nr*nrIn], &output);
		// compare
		e = result[nr] - output;
		// average error over time
		error += 0.01 * (fabs(e) - error);
		// train
		net.bprop(&e);
        if (doPrint)
        {
            std::cout << "av. error " << error << " out " << output << "           \r";
            //net.dump();
        }
        ++count;
	}
    if (doPrint)
    {
        std::cout << "took "<<count<<" epochs                                      \n";

        std::cout << "testing...\n";
    }
	error = 0.0;
	for (size_t i=0;i<nrSamples;i++)
	{
		net.fprop(&input[i*nrIn], &output);
		// compare
		e = fabs(result[i] - output);
		error += e;
        if (doPrint)
        {
            for (size_t j=0;j<nrIn;j++)
                std::cout << input[i*nrIn+j] << ", ";
            std::cout << " =\t" << output << " (abs. error " << e << ")\n";
        }
	}
	error /= nrSamples;
    if (doPrint)
        std::cout << "average abs. error " << error << "\n";
    return std::pair<Float, size_t>(error, count);

}

template <typename Float, class Net>
void simple_test_mean(Net& net, size_t nrIn, size_t nrSamples,
                 std::vector<Float>& input, std::vector<Float>& result)
{
    auto tup = simple_test(net, nrIn, nrSamples, input, result),
         av = tup, mi = tup, ma = tup;
    int num = 300;
    std::cout << "testing for " << num << " runs" << std::endl;
    for (int i = 0; i < num; ++i)
    {
        tup = simple_test(net, nrIn, nrSamples, input, result, false);
        av = std::pair<Float, size_t>(av.first + tup.first, av.second + tup.second);
        mi = std::pair<Float, size_t>(std::min(mi.first, tup.first),
                                      std::min(mi.second, tup.second));
        ma = std::pair<Float, size_t>(std::max(ma.first, tup.first),
                                      std::max(ma.second, tup.second));
    }
    av.first /= num;
    av.second /= num;
    std::cout << "epochs/error over " << num << " runs:"
              << "\naverage: " << av.second << "\t" << av.first
              << "\nmin:     " << mi.second << "\t" << mi.first
              << "\nmax:     " << ma.second << "\t" << ma.first
              << std::endl;
}


template <typename Float, class Net>
void test_some_pattern(Net& net)
{
	const size_t nrIn = 3;
	const size_t nrSamples = 5;
	std::vector<Float> input(
	{ 	0.0, 0.0, 0.0,
		0.2, 0.3, 0.2,
		0.7, 0.6, 0.5,
		0.9, 0.8, 0.9,
		0.1, 0.5, 0.9
	} );
	std::vector<Float> result(
	{
		0.0,
		0.23,
		0.6,
		0.86,
		0.5
	});
    simple_test_mean(net, nrIn, nrSamples, input, result);
};

template <typename Float, class Net>
void test_xor_pattern(Net& net)
{
    const size_t nrIn = 2;
	const size_t nrSamples = 4;
	std::vector<Float> input(
    { 	0.1, 0.1,
        0.1, 0.9,
        0.9, 0.1,
        0.9, 0.9
	} );
	std::vector<Float> result(
	{
		0.1,
		0.9,
		0.9,
		0.1
	});
    simple_test_mean(net, nrIn, nrSamples, input, result);
};







template <typename Float, class Net>
void xor_test(Net& net)
{

	const size_t nrIn = 2;
	const size_t nrSamples = 4;
	const Float X = 0.9;
	const Float O = 0.1;
	std::vector<Float> xor_input(
	{
		O, O,
		O, X,
		X, O,
		X, X
	});
	std::vector<Float> xor_result(nrSamples);

	for (size_t i=0;i<nrSamples;i++)
	{
		size_t k = (xor_input[i*nrIn]>0.5);
		for (size_t j=1;j<nrIn;j++)
			k = k xor (xor_input[i*nrIn+j]>0.5);
		xor_result[i] = (k)? X : O;
	}

	Float output;

	net.resize(nrIn, 1);
	net.brainwash();
	net.info();
	net.dump();

	// stochastic gradient descent
	std::cout << "training "<<nrIn<<"-XOR pattern...\n";
	Float error = 1.0, e;
	size_t count = 0;

	while (error > 0.01 && count<100000)
	{
		// choose pattern
		size_t nr = rand()%nrSamples;
		// feed forward
		net.fprop(&xor_input[nr*nrIn], &output);
		// compare
		e = xor_result[nr] - output;
		// average error over time
		error += 0.01 * (fabs(e) - error);
		// train
		net.bprop(&e);
		std::cout << "av. error " << error << " out " << output << "           \r";
		//net.print();
		count++;
	}
	std::cout << "took "<<count<<" epochs                                      \n";

	std::cout << "testing...\n";
	error = 0.0;
	for (size_t i=0;i<nrSamples;i++)
	{
		net.fprop(&xor_input[i*nrIn], &output);
		// compare
		e = fabs(xor_result[i] - output);
		error += e;
		for (size_t j=0;j<nrIn;j++)
			std::cout << xor_input[i*nrIn+j] << ", ";
		std::cout << " =\t" << output << " (abs. error " << e << ")\n";
	}
	error /= nrSamples;
	std::cout << "average abs. error " << error << "\n";
}







template <typename F>
void maint()
{
    //MNN::Perceptron<float, MNN::Activation::Linear> net;

    MNN::StackSerial<F> net;
    auto l1 = new MNN::Perceptron<F, MNN::Activation::Tanh>(10,10, 1);
    auto l2 = new MNN::Perceptron<F, MNN::Activation::Logistic>(10,10, 1);
    auto l3 = new MNN::Perceptron<F, MNN::Activation::Linear>(10,10, 0.1);

    l1->setMomentum(.5);
    //l1->setLearnRateBias(0.01);
    l2->setMomentum(.5);
    l3->setMomentum(.5);

    net.add(l1);
    net.add(l2);
    net.add(l3);

    test_xor_pattern<F>(net);
}

template <typename F>
void printState(const F* state, size_t width, size_t height, std::ostream& out = std::cout)
{
    for (size_t j=0; j<height; ++j)
    {
        for (size_t i=0; i<width; ++i, ++state)
        {
            out << *state << " ";
        }
        out << std::endl;
    }
}

template <typename F>
void printStateAscii(const F* state, size_t width, size_t height, std::ostream& out = std::cout)
{
    for (size_t j=0; j<height; ++j)
    {
        for (size_t i=0; i<width; ++i, ++state)
        {
            out << ( *state > .7 ? '#' : *state > .35 ? '*' : '.' );
        }
        out << std::endl;
    }
}




template <typename Net>
void evaluateRbm(Net rbm, const MnistSet& set)
{
    struct Sample
    {
        size_t index;
        float err_cd, err_rec;
        std::string img;
    };

    std::vector<Sample> samples;
    for (size_t i = 0; i < set.numSamples(); ++i)
    {
        Sample s;
        s.index = i;
        s.err_cd = rbm->cd(set.image(i), 3, 0.);
        s.err_rec = rbm->compareInput(set.image(i));
        std::stringstream strs;
        printStateAscii(rbm->input(), set.width(), set.height(), strs);
        s.img = strs.str();

        samples.push_back(s);
    }

    std::sort(samples.begin(), samples.end(),
              [](const Sample& l, const Sample& r)
    {
        return l.err_rec < r.err_rec;
        //return l.err_cd < r.err_cd;
    });

    for (const auto& s : samples)
    {
        std::cout << "image #" << s.index
                  << ", cd " << s.err_cd
                  << ", rec " << s.err_rec
                  << "\n" << s.img;
        //printState(rbm->output(), rbm->numOut(), 1);
        std::cout << std::endl;

        std::cin.get();
    }
}

template <typename F>
void testRbm()
{
#define MNIST

#ifdef MNIST
    MnistSet set;
    set.load("/home/defgsus/prog/DATA/mnist/t10k-labels.idx1-ubyte",
             "/home/defgsus/prog/DATA/mnist/t10k-images.idx3-ubyte");
    size_t numIn = set.width() * set.height();

//    printStateAscii(set.image(0), set.width(), set.height());
#else
    size_t numIn = 10;
    std::vector<F> input(numIn);
    for (auto& f : input)
        f = MNN::rnd(F(0), F(1));
#endif

    auto rbm = new MNN::Rbm<F, MNN::Activation::Logistic>(numIn, 20, 1);
    rbm->brainwash();
    rbm->setMomentum(0.5);

    /*
    std::fstream fs;
    fs.open("rbm.txt", std::ios_base::out);
    rbm->serialize(fs);
    fs.close();

    rbm->resize(1, 1);
    */
#if 1
    // load previous
    {
        std::fstream fs;
        fs.open("mnist_rbm_20h.txt", std::ios_base::in);
        rbm->deserialize(fs);
        fs.close();
        //rbm->dump();
    }

    evaluateRbm(rbm, set);
    return;
#endif

    rbm->info();

    F err = 0., err_sum = 0., err_min = 0., err_max = 0.;
    size_t err_count = 1;
    const size_t num = 250000;
    for (size_t it=0; it<num; ++it)
    {
        if (it % 1000 == 0)
        {
            //rbm->dump();
            std::cout << "step " << std::left << std::setw(9) << it
                      << " err " << std::setw(9) << err_min
                      << " - " << std::setw(9) << err_max
                      << " av " << std::setw(9) << (err_sum / err_count)
                      << " avweight " << rbm->getWeightAverage()
                      << std::endl;

            err_max = 0.;
            err_min = -1.;
        }

#ifdef MNIST
        size_t idx = rand() % set.numSamples();
        //idx = idx % 10;
        err = rbm->cd(set.image(idx), 5, 0.01);
        err = rbm->compareInput(set.image(idx));
#else
        err = rbm->cd(&input[0], 2, 0.1);
#endif
        // gather error stats
        if (err_min < 0.)
            err_min = err;
        else
            err_min = std::min(err_min, err);
        err_max = std::max(err_max, err);

        if (err > 0.)
        {
            err_sum += err;
            ++err_count;
        }

        if (err > 0.0 && err < 20.)
        {
            //printStateAscii(rbm->input(), set.width(), set.height());
            //printStateAscii(rbm->output(), rbm->numOut(), 1);
        }
    }

#if 0
    {
        std::fstream fs;
        fs.open("mnist_rbm_20h.txt", std::ios_base::out);
        rbm->serialize(fs);
        fs.close();
    }
#endif

}



int main()
{
	srand(time(NULL));

    //TrainPosition t; t.exec(); return 0;
    //TrainMnist t; t.exec(); return 0;

    maint<double>();
    //testRbm<float>();

	return 0;
}
