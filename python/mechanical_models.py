from abc import ABC, abstractmethod
import numpy as np
from scipy import constants as const
from scipy.special import gamma
import mpmath

class Diagram(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @property
    @abstractmethod
    def width(self):
        pass

    @property
    @abstractmethod
    def height(self):
        pass

    @staticmethod
    def str2array(s):
        lines_lengths =  list(map(len, s.splitlines()))
        maxl = max(lines_lengths)
        assert lines_lengths[-2] == maxl, """connection must be the longest line"""
        return np.array([
            [
                c
                for c in line.ljust(maxl)
            ]
            for line in s.splitlines()
        ])

    def __add__(self, other):
        """self and other in series"""
        return SeriesDiagram(self, other)

    def __mul__(self, other):
        """self and other in parallel"""
        return ParallelDiagram(self, other)

class DiagramLeaf(Diagram):
    """A single component in the diagram"""

    def __str__(self):
        return "\n".join(["".join(line) for line in self.array])

    @property
    def width(self):
        return self.array.shape[1]

    @property
    def height(self):
        return self.array.shape[0]

class Spring(DiagramLeaf):
    """A spring of modulus G."""
    def __init__(self, G):
        self.array = Diagram.str2array(f"""

____╱╲  ╱╲  ╱╲  ___{'_'*len(G)}
      ╲╱  ╲╱  ╲╱ {G}
""")

class Dashpot(DiagramLeaf):
    """A dashpot of viscosity eta."""
    def __init__(self, eta):
        self.array = Diagram.str2array(f"""
    ___
_____| |____{'_'*len(eta)}
    _|_| {eta}
""")

class Springpot(DiagramLeaf):
    """A springpot of exponent alpha and quasiproperty V."""
    def __init__(self, V, alpha):
        self.array = Diagram.str2array(f"""

____╱╲____{'_'*(len(V)+len(alpha))}
    ╲╱ {V}, {alpha}
""")

class SeriesDiagram(Diagram):
    """Two or more elements in series"""
    def __init__(self, *args):
        assert(len(args)>1), f"Only {len(args)} elements were provided."
        self.children = []
        for child in args:
            if isinstance(child, SeriesDiagram):
                self.children += child.children
            else:
                self.children.append(child)
    @property
    def width(self):
        return sum(child.width for child in self.children)

    @property
    def height(self):
        return max(child.height for child in self.children)

    def __str__(self):
        ret = ['' for l in range(self.height)]
        for child in self.children:
            for l, line in enumerate(f'{child}'.splitlines()[::-1]):
                ret[len(ret)-l-1] += line
        return "\n".join(ret)

class ParallelDiagram(Diagram):
    """Two or more elements in parallel"""
    def __init__(self, *args):
        assert(len(args)>1), f"Only {len(args)} elements were provided."
        self.children = []
        for child in args:
            if isinstance(child, ParallelDiagram):
                self.children += child.children
            else:
                self.children.append(child)
    @property
    def width(self):
        return max(child.width for child in self.children)+8

    @property
    def height(self):
        return sum(child.height for child in self.children)

    def __str__(self):
        maxL = self.width-8
        ret = []
        lines = f'{self.children[0]}'.splitlines()
        for line in lines[:-2]:
            ret.append(line.center(maxL+8))
        ret.append(' '*4 + lines[-2].center(maxL, '_')+' '*4)
        ret.append('   |'+lines[-1].center(maxL)+'|   ')
        for child in self.children[1:-1]:
            lines = f'{self.child}'.splitlines()
            for line in lines[:-2]:
                ret.append('   |'+line.center(maxL)+'|   ')
            ret.append('   |'+lines[-2].center(maxL, '_')+'|   ')
            ret.append('   |'+lines[-1].center(maxL)+'|   ')
        lines = f'{self.children[-1]}'.splitlines()
        for line in lines[:-2]:
            ret.append('   |'+line.center(maxL)+'|   ')
        ret.append('___|'+lines[-2].center(maxL, '_')+'|___')
        ret.append(lines[-1].center(maxL+8))
        return '\n'.join(ret)

def text_spring(G):
    return f"""

____╱╲  ╱╲  ╱╲  ___{'_'*len(G)}
      ╲╱  ╲╱  ╲╱ {G}
"""

def text_dashpot(eta):
    return f"""
    ___
_____| |____{'_'*len(eta)}
    _|_| {eta}
"""

def text_springpot(V, alpha):
    return f"""

____╱╲____{'_'*(len(V)+len(alpha))}
    ╲╱ {V}, {alpha}
"""

def text_series(a,b):
    """join two element strings in series"""
    maxla = max(map(len, a.splitlines()))
    alines = [line.ljust(maxla) for line in a.splitlines()]
    blines = [line for line in b.splitlines()]
    if len(alines)>len(blines):
        blines = ['']*(len(alines) - len(blines)) + blines
    elif len(blines)>len(alines):
        alines = [' '*maxla]*(len(blines) - len(alines)) + alines
    ret = [
        aline+bline
        for aline, bline in zip(alines, blines)
    ]
    return '\n'.join(ret)

def text_parallel(*args):
    """join several element strings in parallel"""
    assert(len(args)>1), """At least 2 elements can be in parallel"""
    maxls = [
        max(map(len, a.splitlines()))
        for a in args
        ]
    maxL = max(maxls)
    ret = []
    for line in args[0].splitlines()[:-2]:
        ret.append(line.ljust(maxls[0]).center(maxL+8))
    line = args[0].splitlines()[-2]
    ret.append('    '+line.ljust(maxls[0],'_').center(maxL, '_')+'    ')
    line = args[0].splitlines()[-1]
    ret.append('   |'+line.ljust(maxls[0]).center(maxL)+'|   ')
    for a in args[1:-1]:
        maxl = max(map(len, a.splitlines()))
        for line in a.splitlines()[:-2]:
            ret.append('   |'+line.ljust(maxl).center(maxL)+'|   ')
        line = a.splitlines()[-2]
        ret.append('   |'+line.ljust(maxl,'_').center(maxL, '_')+'|   ')
        line = a.splitlines()[-1]
        ret.append('   |'+line.ljust(maxl).center(maxL)+'|   ')
    for line in args[-1].splitlines()[:-2]:
        ret.append('   |'+line.ljust(maxls[-1]).center(maxL)+'|   ')
    line = args[-1].splitlines()[-2]
    ret.append('___|'+line.ljust(maxls[-1], '_').center(maxL, '_')+'|___')
    line = args[-1].splitlines()[-1]
    ret.append(line.ljust(maxls[-1]).center(maxL+8))
    return '\n'.join(ret)

class LinearMechanicalModel(ABC):
    diagram = """No implemented diagram"""
    @abstractmethod
    def Laplace_G(self, s):
        """The modulus in the Laplace domain.

        In principle, only this method needs to be implemented as all other
        methods can be derived from the results of this one. However in practice,
        analytical simplified expressions of the other methods will be more
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

    def msd(self, t, T, a, d=3):
        """Mean square displacement function of time of a particle of radius a (m) immersed in the
        medium at temperature T (°C). Dimensionality is d."""
        return d*const.Boltzmann * const.convert_temperature(T, 'C', 'K') /(3*np.pi*a) * self.J(t)


class Elastic(LinearMechanicalModel):
    """Elastic solid of constant elasticity G (Pa)"""

    diagram = text_spring('G')

    def __init__(self, G):
        self.G = G

    def Laplace_G(self, s):
        return np.full_like(s, self.G)

    def Gp(self, ω):
        return np.full_like(ω, self.G)

    def Gpp(self, ω):
        return np.zeros_like(ω)

    def tandelta(self, ω):
        return np.zeros_like(ω)

    def J(self, t):
        return np.full_like(t, 1/self.G)


class Newtonian(LinearMechanicalModel):
    """Newtonian fluid of constant viscosity η (Pa.s)"""

    diagram = text_dashpot('η')

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

class Maxwell(LinearMechanicalModel):
    """Maxwell model of an elasticity G (Pa) in series with a viscosity η (Pa.s). The characteristic time is τ (s)"""

    diagram = text_series(text_spring('G'), text_dashpot('η'))

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

class KelvinVoigt(LinearMechanicalModel):
    """Kelvin-Voigt model of an elasticity G (Pa) in parallel to a viscosity η (Pa.s). The characteristic time is τ (s)"""

    diagram = text_parallel(text_spring('G'), text_dashpot('η'))

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

    diagram = text_parallel(text_dashpot('ηs'), Maxwell.diagram)

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

class PowerLaw(LinearMechanicalModel):
    """A springpot element of exponent α and pseudo-property V (Pa.s^α)"""

    diagram = text_springpot('V', 'α')

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

class FractionalMaxwell(LinearMechanicalModel):
    """Two springpot elements in series of respective exponent α and β and respective pseudo-property V (Pa.s^α) and G (Pa.s^β)"""

    diagram = text_series(text_springpot('V', 'α'), text_springpot('G', 'β'))

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
