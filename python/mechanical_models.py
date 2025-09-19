from abc import ABC, abstractmethod
import numpy as np
from scipy.special import gamma
import mpmath

class MechanicalModel(ABC):
    @abstractmethod
    def Laplace_G(self, s):
        """The modulus in the Laplace domain.

        In principle, only this method needs to be implemented as all other
        methods can be derived from the results of this one. However in practice,
        analytic simplified expressions of the other methods will be more
        efficient, in particular J(t) that implies inverse Laplace transform
        that is slow numerically."""
        pass

    def Gp(self, ω):
        """Elastic modulus function of pulsation ω"""
        return np.real(self.Laplace_G(1j * ω))

    def Gpp(self, ω):
        """Viscous modulus function of pulsation ω"""
        return np.imag(self.Laplace_G(1j * ω))

    def tandelta(self, ω):
        """Loss tangent function of pulsation ω"""
        G = self.Laplace_G(1j * ω)
        return np.imag(G) / np.real(G)

    def J(self, t):
        """Creep compilance function of time"""
        Laplace_J = lambda s: 1/s/self.self.Laplace_G(s)
        return np.array([
            mpmath.invertlaplace(Laplace_J, x)
            for x in t], float)

class Newtonian(MechanicalModel):
    """Newtonian fluid of constant viscosity η (Pa.s)"""

    def __init__(self, η):
        self.η = η

    def Laplace_G(self, s):
        return s * self.η

    def Gp(self, ω):
        return np.zeros_like(ω)

    def Gpp(self, ω):
        return ω * self.η

    def tandelta(self, ω):
        return np.full_like(ω, np.inf)

    def J(self, t):
        return t/self.η

class Maxwell(MechanicalModel):
    """Maxwell model of an elasticity G (Pa) in series with a viscosity η (Pa.s). The characteristic time is τ (s)"""

    def __init__(self, G=None, η=None, τ=None):
        assert G is None or η is None or τ is None, "G, η and τ anre not independent. All three cannot be set."
        if G is not None:
            self.G = G
            if η is not None:
                self.η = η
                self.τ = η / G
            elif τ is not None:
                self.τ = τ
                self.η = G * τ
            else:
                raise KeyError("Either η or τ must be defined")
        elif η is not None and τ is not None:
            self.η = η
            self.τ = τ
            self.G = η / τ
        else:
            raise KeyError("Two among G, η or τ must be defined")

    def Laplace_G(self, s):
        return s * self.η / (1 + s*self.τ)

    def Gp(self, ω):
        return self.G * (
            self.τ**2 * ω**2
        )/(
            1 + self.τ**2 * ω**2
        )

    def Gpp(self, ω):
        return self.G * (
            self.τ * ω
        )/(
            1 + self.τ**2 * ω**2
        )

    def tandelta(self, ω):
        return 1/(self.τ * ω)

    def J(self, t):
        return 1/self.G + t/self.η

class KelvinVoigt(MechanicalModel):
    """Kelvin-Voigt model of an elasticity G (Pa) in parallel to a viscosity η (Pa.s). The characteristic time is τ (s)"""

    def __init__(self, G=None, η=None, τ=None):
        assert G is None or η is None or τ is None, "G, η and τ anre not independent. All three cannot be set."
        if G is not None:
            self.G = G
            if η is not None:
                self.η = η
                self.τ = η / G
            elif τ is not None:
                self.τ = τ
                self.η = G * τ
            else:
                raise KeyError("Either η or τ must be defined")
        elif η is not None and τ is not None:
            self.η = η
            self.τ = τ
            self.G = η / τ
        else:
            raise KeyError("Two among G, η or τ must be defined")

    def Laplace_G(self, s):
        return self.G + s * self.η

    def Gp(self, ω):
        return np.full_like(ω, self.G)

    def Gpp(self, ω):
        return ω * self.η

    def tandelta(self, ω):
        return self.η * ω / self.G

    def J(self, t):
        return (1 - np.exp(-t/self.τ)) / self.G

class JohnsonSegalman(Maxwell):
    """Johnson-Segalmant model of a viscosity ηs (Pa.s) in parallel to a Maxwell of elasticity G (Pa) and viscosity η (Pa.s)."""

    def __init__(self, G, η, ηs):
            self.G = G
            self.η = η
            self.ηs = ηs

    def Laplace_G(self, s):
        return Maxwell.Laplace_G(self, s) + ω * self.ηs

    def Gpp(self, ω):
        return Maxwell.Gpp(self, ω) + ω * self.ηs

    def tandelta(self, ω):
        return Maxwell.tandelta(self, ω) + self.η * ω / Maxwell.Gpp(self, ω)

    def J(self, t):
        return t/(self.η + self.ηs) + 1/self.G * (self.η/(self.η + self.ηs))**2 * (1 - np.exp(-self.G * (1/self.η + 1/self.ηs)*t))

class PowerLaw(MechanicalModel):
    """A springpot element of exponent α and pseudo-property V (Pa.s^α)"""

    def __init__(self, V, α):
            self.V = V
            self.α = α

    def Laplace_G(self, s):
        return self.V * s ** self.α

    def Gp(self, ω):
        return self.V * np.cos(π * self.α / 2) * ω**self.α

    def Gpp(self, ω):
        return self.V * np.sin(π * self.α / 2) * ω**self.α

    def tandelta(self, ω):
        return np.tan(π * self.α / 2)

    def J(self, t):
        return t**self.α / (self.V * gamma(1 + self.α))

class FractionalMaxwell(MechanicalModel):
    """Two springpot elements in series of respective exponent α and β and respective pseudo-property V (Pa.s^α) and G (Pa.s^β)"""

    def __init__(self, α, β, V=None, G=None, τ=None):
        assert G is None or V is None or τ is None, "G, V and τ are not independent. All three cannot be set."
        self.α = α
        self.β = β
        if τ is not None:
            self.τ = τ
            if G is not None and V is None:
                self.G = G
                self.V = G * (np.sin(π*α/2) - np.cos(π*α/2)) / (np.cos(π*β/2) - np.sin(π*β/2)) * τ**(α-β)
            elif V is not None and G is None:
                self.V = V
                self.G = V * (np.cos(π*β/2) - np.sin(π*β/2)) / (np.sin(π*α/2) - np.cos(π*α/2)) * τ**(β-α)
            else:
                raise KeyError("Either V or G must be defined if τ is defined, but not both.")
        else:
            self.G = G
            self.V = V
            self.τ = (V/G * (np.cos(π*β/2) - np.sin(π*β/2)) / (np.sin(π*α/2) - np.cos(π*α/2)))**(1/(α-β))

    def Laplace_G(self, s):
        return self.V * self.G * s**(self.α + self.β) /(self.V * s**self.α + self.G * s**self.β)

    def Gp(self, ω):
        Go = G*ω**β
        Vo = V*ω**α
        return (
            Go**2 * Vo * np.cos(π * self.α/2) + Vo**2 * Go * np.cos(π * self.β/2)
        )/(
            Vo**2 + Go**2 + 2*Vo*Go*np.cos(π*(self.α-self.β)/2)
        )

    def Gpp(self, ω):
        Go = G*ω**β
        Vo = V*ω**α
        return (
            Go**2 * Vo * np.sin(π * self.α/2) + Vo**2 * Go * np.sin(π * self.β/2)
        )/(
            Vo**2 + Go**2 + 2*Vo*Go*np.cos(π*(self.α-self.β)/2)
        )

    def tandelta(self, ω):
        Go = G*ω**β
        Vo = V*ω**α
        return (
            Go * np.sin(π * self.α/2) + Vo * np.sin(π * self.β/2)
        )/(
            Go * np.cos(π * self.α/2) + Vo * np.cos(π * self.β/2)
        )

    def J(self, t):
        return t**self.α / (self.V*gamma(1+self.α)) + t**self.β/(self.G * gamma(1+self.β))
