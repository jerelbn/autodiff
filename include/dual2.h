// Description: Class for scalar, dual numbers with operator overloads
//              to enable forward-mode automatic differentiation
// Author: Jerel Nielsen
// Date: 20 Sep 2019
#pragma once

#include <cmath>
#include <iostream>

namespace dual2
{

template<typename T = double>
class Dual2
{
public:
    using value_type = T;
    Dual2() : value(T(0)), derivative(T(0)) {}
    Dual2(const T& _value) : value(_value), derivative(T(0)) {}
    Dual2(const T& _value, const T& _derivative) : value(_value), derivative(_derivative) {}
    Dual2(const Dual2& other)
    {
        value = other.value;
        derivative = other.derivative;
    }
    ~Dual2() {}

    void inline setValue(const T& _value) { value = _value; }
    void inline setDerivative(const T& _derivative) { derivative = _derivative; }
    const inline T v() const { return value; }
    const inline T dv() const { return derivative; }

    template<typename E>
    Dual2<T>& operator=(const E& expr) {
        value = expr.v();
        derivative = expr.dv();
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const Dual2<T>& num)
    {
        os << "(" << num.value << ", " << num.derivative << ")";
        return os;
    }

private:
    T value; // Regular number value
    T derivative; // Derivative at value
};


// Binary expression classes
template<typename L, typename R>
class Add
{
    const L& l;
    const R& r;
public:
    using value_type = typename R::value_type;
    Add(const L& _l, const R& _r) : l(_l), r(_r) {}
    value_type v() const { return l.v() + r.v(); }
    value_type dv() const { return l.dv() + r.dv(); }
    const R eval() const { return R(v(), dv()); }
};

template<typename L, typename R>
class Subtract
{
    const L& l;
    const R& r;
public:
    using value_type = typename R::value_type;
    Subtract(const L& _l, const R& _r) : l(_l), r(_r) {}
    value_type v() const { return l.v() - r.v(); }
    value_type dv() const { return l.dv() - r.dv(); }
    const R eval() const { return R(v(), dv()); }
};

template<typename L, typename R>
class Multiply
{
    const L& l;
    const R& r;
public:
    using value_type = typename R::value_type;
    Multiply(const L& _l, const R& _r) : l(_l), r(_r) {}
    value_type v() const { return l.v() * r.v(); }
    value_type dv() const { return l.v() * r.dv() + l.dv() * r.v(); }
    const R eval() const { return R(v(), dv()); }
};

template<typename L, typename R>
class Divide
{
    const L& l;
    const R& r;
public:
    using value_type = typename R::value_type;
    Divide(const L& _l, const R& _r) : l(_l), r(_r) {}
    value_type v() const { return l.v() / r.v(); }
    value_type dv() const { return (l.dv() * r.v() - l.v() * r.dv()) / (r.v() * r.v()); }
    const R eval() const { return R(v(), dv()); }
};

// Unary expression classes
template<typename R>
class Sine
{
    const R& r;
public:
    using value_type = typename R::value_type;
    Sine(const R& _r) : r(_r) {}
    value_type v() const { return sin(r.v()); }
    value_type dv() const { return r.dv() * cos(r.v()); }
    const R eval() const { return R(v(), dv()); }
};

template<typename R>
class Cosine
{
    const R& r;
public:
    using value_type = typename R::value_type;
    Cosine(const R& _r) : r(_r) {}
    value_type v() const { return cos(r.v()); }
    value_type dv() const { return -r.dv() * sin(r.v()); }
    const R eval() const { return R(v(), dv()); }
};

template<typename R>
class Exponential
{
    const R& r;
public:
    using value_type = typename R::value_type;
    Exponential(const R& _r) : r(_r) {}
    value_type v() const { return exp(r.v()); }
    value_type dv() const { return r.dv() * exp(r.v()); }
    const R eval() const { return R(v(), dv()); }
};

template<typename R>
class Logarithm
{
    const R& r;
public:
    using value_type = typename R::value_type;
    Logarithm(const R& _r) : r(_r) {}
    value_type v() const { return log(r.v()); }
    value_type dv() const { return r.dv() / r.v(); }
    const R eval() const { return R(v(), dv()); }
};

template<typename R>
class Power
{
    const R& r;
    const typename R::value_type& k;
public:
    using value_type = typename R::value_type;
    Power(const R& _r, const value_type& _k) : r(_r), k(_k) {}
    value_type v() const { return pow(r.v(),k); }
    value_type dv() const { return k * pow(r.v(), k-value_type(1) * r.dv()); }
    const R eval() const { return R(v(), dv()); }
};

template<typename R>
class Absolute
{
    const R& r;
public:
    using value_type = typename R::value_type;
    Absolute(const R& _r) : r(_r) {}
    value_type v() const { return abs(r.v()); }
    value_type dv() const { return r.dv() * r.v() / abs(r.v()); }
    const R eval() const { return R(v(), dv()); }
};


// Operator overloads
template <typename L, typename R>
Add<L, R> operator+(const L& l, const R& r) {
    return Add<L, R>(l, r);
}

template <typename L, typename R>
Subtract<L, R> operator-(const L& l, const R& r) {
    return Subtract<L, R>(l, r);
}

template <typename L, typename R>
Multiply<L, R> operator*(const L& l, const R& r) {
    return Multiply<L, R>(l, r);
}

template <typename L, typename R>
Divide<L, R> operator/(const L& l, const R& r) {
    return Divide<L, R>(l, r);
}

template<typename R>
Sine<R> sin(const R& r)
{
    return Sine<R>(r);
}

template<typename R>
Cosine<R> cos(const R& r)
{
    return Cosine<R>(r);
}

template<typename R>
Exponential<R> exp(const R& r)
{
    return Exponential<R>(r);
}

template<typename R>
Logarithm<R> log(const R& r)
{
    return Logarithm<R>(r);
}

template<typename R>
Power<R> pow(const R& r, const typename R::value_type& k)
{
    return Power<R>(r,k);
}

template<typename R>
Absolute<R> abs(const R& r)
{
    return Absolute<R>(r);
}

} // namespace dual2