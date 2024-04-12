# LocalizedPatternsGS.jl
Julia code for computer-assisted proofs of localized patterns in the Gray-Scott system of PDEs

# Computer-assisted proofs of localized patterns and branches of periodic solutions in the 2D Gray-Scott system of PDEs.



Table of contents:


* [Introduction](#introduction)
* [The Gray-Scott system of equations](#the-gray-scott-system-of-equations)
   * [Proof of a branch of periodic solutions limiting a localized pattern](#proof-of-a-branch-of-periodic-solutions-limiting-a-localized-pattern)
* [Utilisation and References](#utilisation-and-references)
* [License and Citation](#license-and-citation)
* [Contact](#contact)



# Introduction

This Julia code is a complement to the article 

#### [[1]](To appear) : "The 2D Gray-Scott model: constructive proofs of existence of stationary localized patterns", M. Cadiot and D. Blanco, [ArXiv Link](To appear).

It provides the necessary rigorous computations of the bounds presented along the paper. The computations are performed using the package [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl). The mathematical objects (spaces, sequences, operators,...) are built using the package [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl) in combination with the add on [D4Fourier](https://github.com/dominicblanco/D4Fourier.jl)


# The Gray-Scott system of equations

The Gray-Scott system of equations
$$\lambda_1 \Delta u_1 -  u_1 + (u_2 + 1 - \lambda_1 u_1)u_1^2 =0$$
$$\Delta u_2 - \lambda_2 u_2 + (\lambda_2 \lambda_1 - 1)u_1 = 0$$
is known to have localized solutions on $\mathbb{R}^2$ that vanish at infinity. These solutions are called localized patterns (see [[1]](https://arxiv.org/abs/2302.12877) for an introduction to the subject). We perform the analysis to prove these patterns in Section 4 of [[1]](To appear).

In the case $\lambda_1 \lambda_2 = 1$, the system reduces to a scalar equation
$$\lambda_1 \Delta u_1 - u_1 + (1 - \lambda_1 u_1)u_1^2$$
which is also known to have localized patterns. We perform the analysis to prove these patterns in Section 5 of [[1]](To appear).

## Proof of a branch of periodic solutions limiting a localized pattern

In Section 4.4 in [[1]](To appear), we prove that, under the condition (79) in Theorem 4.11, localized patterns can be proven to be the limit of a branch of (spatially) periodic solutions as the period tends to infinity for the Gray-Scott system of equations. In particular this condition involves the explicit computation of multiple bounds, which is achieved in the present code. Moreover, we verify that condition (79) is verified for three different localized patterns and we obtain a constructive proof of existence of a branch of periodic solutions limiting the localized pattern. By using [D4Fourier](https://github.com/dominicblanco/D4Fourier.jl), we obtain proofs of the $D_4$-symmetry. Furthermore, one of the patterns we prove is a non-radial localized pattern, the first of its kind found in Gray-Scott. The other patterns appear to possess radial symmetry, but this symmetry is not proven. Then, Theorem 5.8 in [[1]](To appear) provides the same result for the scalar case $\lambda_2 \lambda_1 = 1$.

We provide as well candidate solutions for the proofs, which are given in the files U0_spikeaway.jld2, U0_ring.jld2, and U0_leaf.jld2. These correspond to the sequence $U_0$ in Section 3.2 representing the approximate solution $u_0$. In particular, $U_0$ has already been projected in the set of sequences representing trace zero functions (see Section 3.2). Consequently, the Fourier series associated to $U_0$ represents a smooth function on $\mathbb{R}^2$ with compact support on a square $\Omega_0$. In the scalar case, the approximate solution is computed in the code using a Newton method.

Given these approximate solution, LocalizedPatternsGS.jl provides the explicit computation of the bounds in Lemmas 4.3, 4.5, and 4.6. It also provides a value for $r_0$ where the proof is successful. In particular, the theorem states that there exists a smooth curve 
$$\{\tilde{u}(q) : q \in [d,\infty]\} \subset C^\infty(\mathbb{R}^2) \times C^\infty(\mathbb{R}^2)$$
such that $\tilde{u}(q)$ is a periodic solution to the Gray-Scott system of PDEs with period $2q$ in both variables.  In particular, $\tilde{u}(\infty)$ is a localized pattern on $\mathbb{R}^2.$ Finally, the value of $r_0$ provides a uniform control on the branch of periodic solutions, making the proof constructive. 

Then, the code LocalizedPatternsReducedGS.jl provides the same results for the scalar case. In particular, it computes the bounds in Lemmas 5.3, 5.4, and 5.5. It also provides a value for $s_0$ where the proof is successful. In particular, the theorem states that there exists a smooth curve 
$$\{\tilde{u}(q) : q \in [d,\infty]\} \subset C^\infty(\mathbb{R}^2)$$
such that $\tilde{u}(q)$ is a periodic solution to the Gray-Scott reduced PDE with period $2q$ in both variables.  In particular, $\tilde{u}(\infty)$ is a localized pattern on $\mathbb{R}^2.$ Finally, the value of $s_0$ provides a uniform control on the branch of periodic solutions, making the proof constructive. 

 
 # Utilisation and References

 The codes in LocalizedPatternsGS.jl and LocalizedPatternsReducedGS.jl can serve to prove other patterns than the one provided as illustration should one have the numerical candidates and they possess $D_4$-symmetry. In particular, the projection in the set of functions with null trace of order $2$ is computed in the code, meaning one can just provide the numerical candidate and attempt a proof.
 
 The code is build using the following packages :
 - [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl) 
 - [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl)
 - [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
 - [IntervalLinearAlgebra](https://github.com/JuliaIntervals/IntervalLinearAlgebra.jl)
 - [JLD2](https://github.com/JuliaIO/JLD2.jl)
 - [D4Fourier](https://github.com/dominicblanco/D4Fourier.jl)
 
 
 # License and Citation
 
This code is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).
  
If you wish to use this code in your publication, research, teaching, or other activities, please cite it using the following BibTeX template:

```
@software{LocalizedPatternSH.jl,
  author = {Matthieu Cadiot and Dominic Blanco},
  title  = {LocalizedPatternsGS.jl},
  url    = {https://github.com/dominicblanco/LocalizedPatternsGS.jl},
  note = {\url{ https://github.com/dominicblanco/LocalizedPatternsGS.jl},
  year   = {2024},
  doi = {10.5281/zenodo.10967034}
}
```
DOI : [10.5281/zenodo.10967034](https://doi.org/10.5281/zenodo.10967034) 


# Contact

You can contact us at :

matthieu.cadiot@mail.mcgill.ca
dominic.blanco@mail.mcgill.ca
