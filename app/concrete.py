from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from scipy.integrate import trapezoid
import pandas as pd
from functools import partial
from app.utils import (
    regula_falsi,
    NegativeSignException,
    PositiveSignException,
    DesignException,
    INFLATION,
)

kPA_TO_KSI = 0.000145038
INCHES_TO_METERS = 0.0254
MomentRotationDataFrame = pd.DataFrame  # columns M and r with index


@dataclass
class Rebar:
    # areas in m2
    num: int = 3
    num2: float = 32e-6
    num3: float = 71e-6
    num4: float = 127e-6
    num5: float = 198e-6
    num6: float = 285e-6
    num8: float = 507e-6
    num10: float = 819e-6
    num14: float = 1452e-6
    num18: float = 2581e-6
    _table: np.ndarray | None = None
    nums: list | None = None
    _allowed_num_bars: list | None = None
    _areas_by_num: dict[int, float] | None = None
    _diameters_by_num: dict[int, float] | None = None

    def __post_init__(self):
        self.nums = [2, 3, 4, 5, 6, 8, 10, 14, 18]
        self._diameters = (
            INCHES_TO_METERS * np.array([2, 3, 4, 5, 6, 8, 10, 14, 18]) / 8
        )
        self._areas_by_num = {
            2: self.num2,
            3: self.num3,
            4: self.num4,
            5: self.num5,
            6: self.num6,
            8: self.num8,
            10: self.num10,
            14: self.num14,
            18: self.num18,
        }
        self._diameters_by_num = {n: d for d, n in zip(self._diameters, self.nums)}
        self._allowed_num_bars = range(4, 12 + 1, 2)
        self._table = np.array(
            [
                [k * num for k in self._allowed_num_bars]
                for num in self._areas_by_num.values()
            ]
        )

    @property
    def area(self) -> float:
        return self._areas_by_num[self.num]

    @property
    def diameter(self) -> float:
        return float(self._diameters_by_num[self.num])

    @classmethod
    def area_to_bars(cls, area: float):
        b = cls()
        i, j = np.unravel_index(
            np.argmin(abs(b._table - area), axis=None), b._table.shape
        )
        caliber = b.nums[i]
        num = b._allowed_num_bars[j]
        real_area = num * b._areas_by_num[caliber]
        diameter = b._diameters_by_num[caliber]
        # print(f"{num=} {caliber=} {real_area=}")
        return real_area, num, caliber, diameter


@dataclass
class RectangularConcreteColumn:
    """
    units: kN - m - s
    ---
    a beam is a special case of a column where P = 0
    simple bernoulli flexure theory

    a simplification of a column where steel is clumped to the top and bottom of the section
    this means no stirrups passing through the middle of the section
    if stirrup spacing varies of height of column, s should reflect spacing in hinge region


    X-section is considered constant accross the length of the member

    same reinforcement steel quality compression/tension fy = fyc = fy
    fyw can be different but all of them are 420 MPa by default

    compression/tension are so defined in the case of positive moment (natural bending of beams)
    thetas = rotation
    phis = curvatures
    eps = unit strains
    """

    # geometric properties
    b: float = 1.0  # width of the compression zone
    h: float = 1.0  # total height of section
    L: float = 1.0  # length of the column
    P: float = 0.0  # axial force P>0 is compression
    N: float = 0.0  # alias for P
    cover: float = 0.03  # assume symmetric covering
    fc: float = 30e3  # concrete compressive strength at 28 days
    fct: float = 3e3  # concrete tensile strength at 28 days := 1/10 fc
    Ast: float | None = None  # area of steel in the tension zone
    Asc: float = 0.0  # area of steel in the compression zone
    Asw: float = 0.0  # area of steel in web
    As: float = 0.0  # As = Asc + Ast
    _rebar: Rebar | None = field(
        default_factory=Rebar
    )  # rebar number to compute advanced parameters
    Asw: float = 0.0  # stirrup/web steel area
    Acc: float | None = None  # area of the confined core
    fy: float = 420e3  # longitudinal steel yield strength
    s: float = 0.2  # stirrup spacing
    _stirrup: Rebar | None = field(
        default_factory=Rebar
    )  # rebar number to compute advanced parameters
    fyw: float = 420e3  # stirrup yield strength
    Es: float = 200e6  # Young's modulus for all steel reinforcement
    Ec: float | None = None  # Young's modulus for concrete
    Ac: float | None = None  # effective cross-sectional area for flexure A = bd
    Ag: float | None = None  # gross cross-sectional area Ag = bh
    d: float | None = (
        None  # distance from the extreme fiber in compression to the centroid of the longitudinal reinforcement on the tension side of the member
    )
    c: float | None = None  # depth of neutral axis
    dp: float | None = (
        None  # distance from the extreme compression fiber to the centroid of the longitudinal compression steel
    )
    p: float | None = None  # longitudinal total reinforcement ratio, p = As/bd.
    pc: float | None = None  # longitudinal compression reinforcement ratio pc = Asc/bd
    pt: float | None = None  # longitudinal tension reinforcement ratio pc = Asc/bd
    pw: float | None = (
        None  # compression ratio of transverse reinforcement, in region of close spacing at column end pw = Asw/sb
    )
    pmin: float | None = None
    pmax: float | None = None
    Asmin: float | None = None
    Asmax: float | None = None
    pcc: float | None = None  # ratio of longitudinal reinforcement to the confined core
    pweff: float | None = (
        None  # effective ratio of transverse reinforcement, in region of close spacing pweff=pw*fy/fc′
    )
    beta: float | None = None
    My: float | None = 0.0
    Mc: float | None = 0.0
    Mu: float | None = 0.0
    Ke: float | None = 0.0
    fpc: float | None = None
    fyMPa: float | None = None
    fcMPa: float | None = None
    fpcMPa: float | None = None
    nu: float | None = None  # axial load ratio P/(Ag fpc)
    pc: float | None = None
    pt: float | None = None
    qt: float | None = None
    qc: float | None = None
    bar: float | None = None
    num_bars: float | None = None
    bar_caliber: float | None = None
    bar_diameter: float | None = None
    bar_num: float | None = None
    bars_top: float | None = None
    bars_bottom: float | None = None
    FRM: float | None = 0.9  # factor of safety for moments, used for design
    FRV: float | None = 0.75  # factor of safety for shears, used for design
    # eps_c : float | None = 0.0015 # assumed compression strain in the concrete
    eps_cu: float = 0.0038  # assumed maximum useable compression strain in the concrete
    eps_y: float = 0.0021  # yield strain of steel fy/Es
    singly_reinforced: bool = False
    Ig: float | None = None  # gross stiffness
    Iy: float | None = None  # (effective) yield stiffness
    length: float = 1.0
    L: float = 1.0  # use length instead of L
    Ls: float | None = (
        None  # shear span, distance between column end and point of inflection, shear span is length of equivalent cantilever
    )
    Lpl: float = 0.03
    DESIGN_TOL: float = 0.05
    EFFECTIVE_INERTIA_COEFF: float | None = None  # Iy = coeff*Ig
    alpha_steel_type: float = (
        0.016  # 0.016 for ductile hot-rolled or heat-treated steel and to 0.0105 for cold-worked steel
    )
    alpha_slippage: int = (
        1  # asl is the zero-one variable for slip, equal to 1 if there is slippage of the longitudinal bars from their anchorage beyond the section of the maximum moment, or to 0 if there is not
    )
    alpha_postyield: float = 0.13
    alpha_confinement: int = 1  # EuroCode confinement effectiveness factor
    alpha_cyclic: int = 1  # 1 if cyclic load, 0 if monotonic
    betaParkAng: float | None = None
    betaJiangCheng: float | None = None
    gammaParkAng: float | None = None
    gammaJiangCheng: float | None = None
    Et: float | None = None  # energy capacity
    theta_y: float | None = None
    theta_y_fardis: float | None = None
    theta_u: float | None = None
    theta_pc_cyclic: float | None = None
    theta_cap_cyclic: float | None = None
    theta_u_cyclic: float | None = None

    def __repr__(self):
        return f"RectangularConcreteColumn {self.b=:.2f} {self.h=:.2f}"

    def __str__(self):
        s = f"""
set Ic {self.Iy}
set theta_y {self.theta_y}
set theta_p {self.theta_cap_cyclic}
set theta_pc {self.theta_pc_cyclic}
set My {self.My}
set lambda {self.gammaJiangCheng}
set alpha {self.alpha_postyield}
set Icrit {self.Icrit}
set stable {self.stable}
    """
        return s

    def debug(self) -> str:
        """
        used when analysis fails
        """
        axial_ratios = self.P / self.P0
        steel_ratios = 0
        return f"{axial_ratios=:.2f} {steel_ratios=:.3f}"

    @property
    def describe(self):
        s = "\n"
        s += f"As: {self.As:.6f} {self.num_bars}#{self.bar_caliber}\n"
        s += "M2 {:.0f} V {:.0f} - pt {:.2f} pc {:.2f} %\n".format(
            self.My, self.VR, self.pt * 100, self.pc * 100
        )
        return s

    @property
    def properties(self):
        return f"My {self.My:.2f} mu {self.ductility_cyclic:.3f} θy {self.theta_y:.4f} θpl {self.theta_cap_cyclic:.4f} θpc {self.theta_pc_cyclic:.4f} θu {self.theta_u_cyclic:.4f} Λ {1./self.betaJiangCheng:.1f} alpha {self.alpha_postyield:.4f}\n"

    def __post_init__(self):
        self.d = self.h - self.cover
        # distance from extreme compression fiber to the centroid of the longitudinal reinforcement on the compression side
        self.perimeter = 2 * (self.b + self.d)
        self.dp = self.cover if not self.dp else self.dp
        self.Ag = self.b * self.h
        self.Ac = self.b * self.d
        self.fyMPa = self.fy * 1e-3
        self.fcMPa = self.fc * 1e-3
        # this assumes NTC element. AIC would use different formulae for the constants, good enough for now.
        # in reality an Element gets instantiated with beta, alpha etc based on a BC, this should have no reference to a bc? or should it be a param?
        self.beta = (
            (1.05 - self.fcMPa / 140 if (1.05 - self.fcMPa / 140) >= 0.65 else 0.65)
            if not self.beta
            else self.beta
        )
        self.fpc = self.beta * self.fc
        self.fpcMPa = self.fpc * 1e-3
        # this also depends highly on aggregate quality and can be considerably more complex, good enough for now.
        self.Ec = 4400 * self.fcMPa**0.5 * 1e3 if self.Ec is None else self.Ec
        self.eta = self.Es / self.Ec
        self.Ig = self.b * self.h**3 / 12
        xcr = self.h / 2
        self.Mcr = self.Ig * self.fct / (self.h - xcr)
        self.phi_cr = self.Mcr / self.Ec / self.Ig
        self.P0 = self.Ag * self.fpc
        # balanced ratios
        self.Asb = (
            self.fpc / self.fy * (6000 * self.beta / (self.fy + 6000)) * self.Ac
        )  # balanced steel area
        self.pb = self.Asb / self.Ac
        self.qb = self.fy / self.fpc * self.pb
        # max and min ratios
        if self.pmin is not None:
            # self.pmin = 2 * 0.7 * np.sqrt(self.fcMPa) / self.fyMPa
            # self.pmin = self.pmin if self.pmin >= 0.004 else 0.004
            self.qmin = self.pmin * self.fy / self.fpc
            self.Asmin = self.pmin * self.Ac
            # this formulae only work if we disregard compression steel!
            self.Mmin = self.Asmin * self.fy * self.d * (1 - 0.5 * self.qmin)
            self.Mmin2 = (
                self.b * self.d**2 * self.fpc * self.qmin * (1 - 0.5 * self.qmin)
            )
        if self.pmax is not None:
            # self.pmax = 0.06
            # self.pmax = self.fc/self.fy * (320/(600+self.fyMPa))
            self.Asmax = self.pmax * self.Ac
            self.qmax = self.pmax * self.fy / self.fpc
            # self.Mmax = self.b * self.d **2  * self.fpc * self.qmax * (1 - 0.5*self.qmax)
            self.Mmax = self.Asmax * self.fy * (1 - self.beta / 3)

        if self.nu:
            self.P = self.nu * self.P0
        else:
            self.nu = self.P / self.P0

        if self.My:
            self.design(self.My, tol=self.My * self.DESIGN_TOL)
        elif any([self.p, self.As, self.Ast, self.pt]):
            if self.p:
                self.As = self.p * self.Ac
                self.pc = self.pt = self.p / 2
                self.Ast = self.pt * self.Ac
                self.Asc = self.pc * self.Ac
                self.As = self.Ast + self.Asc
            elif self.As:
                self.p = self.As / self.Ac
                self.pc = self.pt = self.p / 2
                self.Ast = self.pt * self.Ac
                self.Asc = self.pc * self.Ac
            # singly reinforced
            elif self.Ast:
                self.pt = self.Ast / self.Ac
                self.pc = self.Asc / self.Ac
                self.p = self.pt + self.pc
                self.As = self.Ast + self.Asc
            elif self.pt:
                self.Ast = self.pt * self.Ac
                self.Asc = self.pc * self.Ac
                self.As = self.Ast + self.Asc
                self.pc = self.Asc / self.Ac
                self.p = self.pt + self.pc
            self.My = self.analyze(Ast=self.Ast, Asc=self.Asc)
        else:
            raise DesignException("Provide either My or (p, As, Ast or pt)")

        self.My = float(self.My)
        self.Vy = 2 * self.My / self.length
        self.q = self.p * self.fy / self.fpc
        self.qc = self.pc * self.fy / self.fpc
        self.qt = self.pt * self.fy / self.fpc
        self.Asw = 2 * self._stirrup.area if self._stirrup is not None else self.Asw
        self.pw = self.Asw / self.s / self.b
        self.ex = self.My / self.P if self.P else None
        # each BC has different cracked and yielding inertias
        # CDMX beams = 0.5, cols = 0.7
        if self.EFFECTIVE_INERTIA_COEFF is None:
            self.Iy = 0.75 * (0.1 + self.nu) ** 0.8 * self.Ig
            self.Iy = self.Iy if self.Iy / self.Ig > 0.2 else 0.2 * self.Ig
            self.Iy = self.Iy if self.Iy / self.Ig < 0.6 else 0.6 * self.Ig
        else:
            self.Iy = self.EFFECTIVE_INERTIA_COEFF * self.Ig

        self.__set_capacities()
        self.__set_advanced_properties()
        super().__post_init__()

    def __set_capacities(self):
        if self.p < 0.015:
            self.Vcr = (
                self.FRV
                * (0.2 + 20 * self.p)
                * 0.3
                * self.fcMPa**0.5
                * (self.Ac * 1e6)
                * 1e-3
            )
        else:
            self.Vcr = (
                self.FRV * 0.16 * self.fcMPa**0.5 * (self.Ac * 1e6) * 1e-3
            )  # formula inputs are MPa, mm. output is in N.

        self.Vsr = 0.7 * (self.Asw * self.fyw) * self.d / self.s
        self.VR = self.Vcr + self.Vsr

        # simplified expressions for singly reinforced members
        self.My2 = self.fy * self.Ast * self.d * (1 - self.q)  # JAPAN
        self.My3 = self.fpc * self.b * self.d**2 * 7 / 8 * self.qt  # AIC

        self.phi_y = 1.7 * self.fy / self.h / self.Es  # Fardis' approximate expression
        self.phi_y2 = self.phi_y_fardis = self.fy / (0.7 * self.d * self.Ec)
        # self.cw = 1.384 * self.qt
        # self.Mu1 = self.Ast * self.fy * self.d * (1 - 0.59* self.q)
        # self.Mu2 = 0.9 * self.d * self.Ast * self.fy
        # self.Mu3 = self.d * self.Ast * self.fy * (1 - 0.425 * self.cw)
        # self.phi_U = self.eps_cu / self.cw / self.d

        self.Ls = (
            self.length / 2
        )  # shear span is M/V but for double curvature, member shear is constant and equal to 2M/L

        # assumes that we have a single bar diameter
        self.theta_y_fardis = (
            self.phi_y * self.Ls / 3
            + 0.0025
            + self.alpha_slippage
            * (
                0.25
                * self.eps_y
                * self.bar_diameter
                * self.fyMPa
                / ((self.d - self.dp) * self.fpcMPa**0.5)
            )
        )
        self.theta_y = self.theta_y_fardis if self.theta_y is None else self.theta_y
        # self.theta_y = self.theta_y_fardis
        self.theta_pc = 0.76 * 0.031**self.nu * (0.02 + 40 * self.pw) ** 1.02
        self.theta_pc = self.theta_pc if self.theta_pc < 0.10 else 0.10
        self.theta_pc_cyclic = 0.5 * self.theta_pc
        self.theta_cap = (
            0.1
            * (1 + 0.55 * self.alpha_slippage)
            * 0.16**self.nu
            * (0.02 + 0.40 * self.pw) ** 0.43
            * 0.54 ** (0.01 * self.fpcMPa)
        )
        self.theta_cap_cyclic = 0.7 * self.theta_cap
        self.theta_u = self.theta_y + self.theta_cap + self.theta_pc
        self.ductility = (self.theta_y + self.theta_cap) / self.theta_y
        self.theta_u_cyclic = (
            self.theta_y + self.theta_cap_cyclic + self.theta_pc_cyclic
        )
        # self.theta_u_cyclic = 2 * self.theta_y
        self.ductility_cyclic = (self.theta_y + self.theta_cap_cyclic) / self.theta_y
        self.Ks = self.My / self.theta_y
        self.Mc = (
            self.My + self.alpha_postyield * self.Ks * self.theta_cap_cyclic
        )  # My + delta M

    def __set_advanced_properties(self):
        # the alpha confinement ratio formula depends highly on the stirrup geometry,  let's assume 1-sum(wi)**2/6bh = 0.5
        # this is consistent if two designs have the same geometry
        self.hcc = self.h - 2 * self.cover
        self.bcc = self.b - 2 * self.cover
        self.Acc = self.bcc * self.hcc
        self.pcc = self.As / self.Acc
        # self.alpha_McMy = 1 - 1.25 * 0.89**self.nu * 0.91 ** (0.01 * self.fc)
        # self.Mc = (1 + self.alpha_McMy) * self.My
        self.alpha_confinement = (
            0.5
            * ((1 - self.s) / (2 * self.bcc))
            * ((1 - self.s) / (2 * self.hcc))
            / (1 - self.pcc)
        )
        self.betaJiangCheng = (
            0.818 ** (100 * self.alpha_confinement * self.pw * self.fyw / self.fpc)
            * (0.023 * self.L / self.h + 3.352 * self.nu**2.35)
            + 0.039
        )
        n0 = self.nu if self.nu else 0.2
        n0 = n0 if n0 >= 0.2 else 0.2
        pt = self.pt * 100 if self.pt > 0.0075 else 0.75
        pw = self.pw * 100
        self.betaParkAng = 0.7**pw * (
            0.073 * self.L / self.d + 0.24 * n0 + 0.314 * pt - 0.447
        )
        self.gammaParkAng = 1.0 / self.betaParkAng
        self.gammaJiangCheng = 1.0 / self.betaJiangCheng

        self.Et = self.gammaJiangCheng * self.My * self.theta_cap_cyclic
        self.Icrit = (
            (1 + self.alpha_postyield)
            * self.L
            * self.My
            / 9
            / self.Ec
            / self.theta_pc_cyclic
        )
        self.stable = self.Iy > self.Icrit

    def compute_net_worth(self) -> float:
        # normalized by 1k dollars
        STEEL_DENSITY_TON = 7.85
        STEEL_TON_UNIT_COST = 920
        CONCRETE_M3_UNIT_COST = 150
        # this unit cost is callibrated to give the % of unit costs in real structures
        WORK_UNIT_COST = 1
        num_stirrups = self.L / self.s + 1
        stirrup_volume = num_stirrups * self._stirrup.area * self.perimeter
        longitudinal_volume = self.L * self.As
        # print(f"{stirrup_volume=} {longitudinal_volume=} {self.s=} {num_stirrups=}")
        steel = (
            (stirrup_volume + longitudinal_volume)
            * STEEL_DENSITY_TON
            * STEEL_TON_UNIT_COST
        )
        # concrete costs the same when poured no matter the Q or the fpc
        concrete = self.L * self.Ag * CONCRETE_M3_UNIT_COST
        # there is more work involved when stirrup spacing is smaller i.e. when concrete is more confined
        # when unions/overlapping with beams are stricter it is more work
        work = 0.722 * num_stirrups**2 * WORK_UNIT_COST
        # print(f"{steel=} {concrete=} {work=}")
        dollars = steel + concrete + work
        dollars = INFLATION * dollars / 1e3
        # dollars = (
        #     dollars / 2
        # )  ## there are two IMK springs, so this will only be half the cost. not sure why this gives low values for elements compares to slabs
        print(f"{dollars=}")
        return dollars

    def analyze(self, As: float | None = None, *, Ast=0, Asc=0, P=0, tol=5, iter=20):
        P = P or self.P
        if As is not None:
            Ast = As / 2
            Asc = As / 2

        def imbalance(c, moment=False):
            es = self.eps_cu * (self.d - c) / c
            fs = es * self.Es
            fs = np.clip(fs, 0, self.fy)
            esp = self.eps_cu * (c - self.dp) / c
            fsp = esp * self.Es
            fsp = np.clip(fsp, 0, self.fy)
            Cs = fsp * Asc
            Cc = self.beta * self.b * self.fpc * c
            Ts = fs * Ast
            T = Ts + P
            C = Cc + Cs
            diff = T - C
            if moment:
                return (
                    Ts * (self.d - self.h / 2)
                    + Cc * (self.h / 2 - c / 2)
                    + Cs * (self.h / 2 - self.dp)
                )
            return diff

        c1, c2 = 1e-6, self.d
        # c1, c2 = self.dp, self.d
        c, _ = regula_falsi(
            imbalance,
            c1,
            c2,
            tol=tol,
            iter=iter,
            # exception_msg=f"no solution exists for this combination of steel and axial load, {self.debug()}",
        )
        moment = imbalance(c, moment=True)
        return moment

    def design(self, My: float, tol=1, iter=100) -> float:
        # balancing_steel = self.P / self.fy / self.Ag
        pmin = self.pmin or 0.005
        # pmin = self.pmin or 4 * Rebar.num3
        pmax = self.pmax or 0.1
        Asmin, Asmax = self.Asmin or pmin * self.Ag, self.Asmax or pmax * self.Ag
        try:
            self.Mymin = self.analyze(Asmin, tol=tol, iter=iter)
        except DesignException as e:
            raise DesignException(f"{pmin=:.3f} is too little {e}")
        try:
            self.Mymax = self.analyze(Asmax, tol=tol, iter=iter)
        except DesignException as e:
            raise DesignException(f"{pmax=:.3f} is too much {e}")

        def target(My, As) -> float:
            Myi = self.analyze(As, tol=tol, iter=iter)
            return Myi - My

        try:
            area, _ = regula_falsi(
                partial(target, My),
                Asmin,
                Asmax,
                tol=tol,
                iter=iter,
                # exception_msg=f"My greater than {self.Mymax=:.1f}]",
            )
        except NegativeSignException as e:
            print("My too big")
            raise DesignException(e)
        except PositiveSignException:
            # means My is smaller than Mymin
            area = Asmin
            self.My = self.Mymin

        pi = area / self.Ac
        _, num, caliber, diameter = Rebar.area_to_bars(area)
        # if self.pmin and pi < self.pmin:
        #     pi = self.pmin
        self.bar_num = float(num)
        self.bar_caliber = float(caliber)
        self.bar_diameter = float(diameter)
        self.p = float(pi)
        self.pc = self.pt = self.p / 2
        self.As = self.p * self.Ag
        self.Ast = self.Asc = self.p / 2 * self.Ag
        if self.pmax and pi > self.pmax:
            raise DesignException(f"p > pmax ({pi:.3f} > pmax {self.pmax:.3f})")
        return area

    def park_ang_kunnath_DS(
        self,
        df_or_x: MomentRotationDataFrame | float,
        moment_col: str = "M",
        rotation_col: str = "r",
        **kwargs,
    ) -> float:
        if isinstance(df_or_x, float):
            DS = 1
        else:
            area = -trapezoid(df_or_x[moment_col], x=df_or_x[rotation_col])
            theta_max = max(abs(df_or_x[rotation_col]))
            mono = max([theta_max - self.theta_y, 0]) / (
                self.theta_u_cyclic - self.theta_y
            )
            cyclic = area / self.Et
            DS = min([mono + cyclic, 1])
            cap = 100 * area / self.Et
            title = f"{area=:.1f}, {self.Et=:.1f} {cap=:.1f}% -- {mono=:.2f} {cyclic=:.2f} {DS=:.2f}"
        return DS

    def elwood_shear_capacity(
        self, shear_force: float = 0.0, axial_force: float = 0.0
    ) -> float:
        """
        at every instant, the combination of shear/axial may change the capacity
        usually we consider P=constant.
        """
        cap = max(
            3.0 / 100
            + 4 * self.pw
            - 1.0 / 40 * shear_force / 1000 / self.Ag / self.fcMPa**0.5
            - 1.0 / 40 * axial_force / 1000 / self.Ag / self.fcMPa,
            0.01,
        )
        return cap

    def ntc_shear_capacity(self):
        return self.VR

    def aci_shear_capacity(
        self, shear_force: float = 0.0, axial_force: float = 0.0
    ) -> float:
        """
        4.2.3.1
        at every instant, the combination of shear/axial may change the capacity
        usually we consider P=constant.
        """
        knl, lambd = 0.85, 1.0
        MUD_over_VUD_d = 3.0  # 7.34 ASCE41, between 2 and 4
        acol = (
            1.0  # if s / d <= 0.75 else linealmente decreciente hasta cero cuando s/d=1
        )
        d, s = self.d, self.s
        Ag, Av = self.Ag, self.Asw
        fcLE = self.fc
        fytLE = self.fyw * 1.2  # some overstrength
        sqrtFcLE = fcLE**0.5
        cap = (
            # 0.6895
            knl
            * (
                acol * (Av * fytLE * d / s)
                + lambd
                * (0.8 * Ag)
                * (
                    (6 * sqrtFcLE / MUD_over_VUD_d)
                    * (1 + axial_force / (6.0 * Ag * sqrtFcLE)) ** 0.5
                )
            )
        )
        return cap

    def moment_rotation_df(self):
        rots = [
            0,
            self.theta_y,
            self.theta_y + self.theta_cap_cyclic,
            self.theta_u_cyclic,
        ]
        Ms = [0, self.My, self.Mc, self.Mu]
        df = pd.DataFrame(Ms, index=rots)
        return df

    def moment_rotation_figure(self):
        df = self.moment_rotation_df()
        fig = df.plot()
        fig.update_layout(
            xaxis_title="rot (rad)",
            yaxis_title="M (kNm)",
            title_text=f"spring M-rot backbone",
        )
        return fig

    def get_and_set_net_worth(self) -> float:
        # required to override the elasticbeamcolumn method
        pass
