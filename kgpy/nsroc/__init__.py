import typing as typ
import dataclasses
import pathlib
import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import astropy.visualization
import astropy.modeling
import astropy.coordinates
import pandas
from kgpy import mixin, Name, vector, plot
from . import sparcs

__all__ = [
    'Event',
    'sparcs',
]


@dataclasses.dataclass
class Trajectory(mixin.Copyable):
    time_start: astropy.time.Time
    time_mission: u.Quantity
    altitude: u.Quantity
    latitude: u.Quantity
    longitude: u.Quantity
    velocity: vector.Vector3D

    @classmethod
    def from_nsroc_csv(
            cls,
            csv_file: pathlib.Path,
            time_start_col: int = 0,
            time_mission_col: int = 1,
            altitude_col: int = 9,
            latitude_col: int = 10,
            longitude_col: int = 11,
            velocity_ew_col: int = 13,
            velocity_ns_col: int = 14,
            velocity_alt_col: int = 15,
    ):
        df = pandas.read_csv(
            csv_file,
            sep=' ',
            skipinitialspace=True,
            header=None,
            skiprows=1,
        )

        return cls(
            time_start=astropy.time.Time.strptime(str(df[time_start_col].values[0])[:~0], '%y%j%H%M%S'),
            time_mission=df[time_mission_col].values * u.s,
            altitude=(df[altitude_col].values * u.m).to(u.km),
            latitude=df[latitude_col].values * u.deg,
            longitude=df[longitude_col].values * u.deg,
            velocity=vector.Vector3D(
                x=(df[velocity_ew_col].values * (u.m / u.s)).to(u.km / u.s),
                y=(df[velocity_ns_col].values * (u.m / u.s)).to(u.km / u.s),
                z=(df[velocity_alt_col].values * (u.m / u.s)).to(u.km / u.s),
            )
        )

    def __post_init__(self):
        self.update()

    def update(self):
        self._time_apogee_cache = None

    @property
    def time(self) -> astropy.time.Time:
        return self.time_start + self.time_mission

    @property
    def time_apogee(self) -> astropy.time.Time:
        if self._time_apogee_cache is None:
            fit = astropy.modeling.fitting.LinearLSQFitter()
            parabola = astropy.modeling.models.Polynomial1D(degree=2, domain=[5.8e4, 5.9e4])
            mask = self.altitude > 200 * u.km
            parabola = fit(parabola, self.time_mission[mask], self.altitude[mask])
            vertex_x = -parabola.c1 / (2 * parabola.c2)
            self._time_apogee_cache = self.time_start + vertex_x
        return self._time_apogee_cache

    def _interp_quantity_vs_time(self, a: u.Quantity, t: astropy.time.Time):
        interpolator = scipy.interpolate.interp1d(
            x=self.time.to_value('unix'),
            y=a,
            kind='quadratic',
            fill_value='extrapolate',
        )
        return interpolator(t.to_value('unix')) << a.unit

    def altitude_interp(self, t: astropy.time.Time) -> u.Quantity:
        return self._interp_quantity_vs_time(self.altitude, t)

    def latitude_interp(self, t: astropy.time.Time) -> u.Quantity:
        return self._interp_quantity_vs_time(self.latitude, t)

    def longitude_interp(self, t: astropy.time.Time) -> u.Quantity:
        return self._interp_quantity_vs_time(self.longitude, t)

    @property
    def earth_location(self) -> astropy.coordinates.EarthLocation:
        return astropy.coordinates.EarthLocation(
            lat=self.latitude,
            lon=self.longitude,
            height=self.altitude,
        )

    @property
    def sun_alt_az(self) -> astropy.coordinates.SkyCoord:
        sun = astropy.coordinates.get_sun(self.time)
        alt_az = astropy.coordinates.AltAz(obstime=self.time, location=self.earth_location)
        return sun.transform_to(alt_az)

    @property
    def sun_zenith_angle(self):
        return self.sun_alt_az.zen

    def sun_zenith_angle_interp(self, t: astropy.time.Time) -> u.Quantity:
        return self._interp_quantity_vs_time(a=self.sun_zenith_angle, t=t)

    def plot_quantity_vs_time(
            self,
            quantity: u.Quantity,
            quantity_name: str = '',
            ax: typ.Optional[plt.Axes] = None,
            time_start: typ.Optional[astropy.time.Time] = None,
    ):
        if ax is None:
            _, ax = plt.subplots()

        ax = plot.datetime_prep(ax)

        if time_start is not None:
            time = time_start + self.time_mission
        else:
            time = self.time

        with astropy.visualization.quantity_support():
            # with astropy.visualization.time_support(format='isot'):
            ax.plot(
                time.to_datetime(),
                quantity,
                label=quantity_name,
            )

        # ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.1, 0.5))
        # ax.legend()

        return ax

    def plot_apogee(self, ax: plt.Axes):
        line = ax.axvline(x=self.time_apogee.to_datetime(), label='apogee', linestyle='--', linewidth=1)
        return line

    def plot_altitude_vs_time(
            self,
            ax: typ.Optional[plt.Axes] = None,
            time_start: typ.Optional[astropy.time.Time] = None,
    ) -> plt.Axes:
        ax = self.plot_quantity_vs_time(
            quantity=self.altitude,
            quantity_name='altitude',
            ax=ax,
            time_start=time_start,
        )
        ax.set_ylabel('altitude (' + ax.get_ylabel() + ')')
        return ax

    def plot_total_velocity_vs_time(
            self,
            ax: typ.Optional[plt.Axes] = None,
            time_start: typ.Optional[astropy.time.Time] = None,
    ) -> plt.Axes:
        return self.plot_quantity_vs_time(
            quantity=self.velocity.length,
            quantity_name='velocity',
            ax=ax,
            time_start=time_start,
        )

    def plot_altitude_and_velocity_vs_time(
            self,
            ax_altitude: typ.Optional[plt.Axes] = None,
            ax_velocity: typ.Optional[plt.Axes] = None,
    ) -> typ.Tuple[plt.Axes, plt.Axes]:
        if ax_altitude is None:
            _, ax = plt.subplots()

        if ax_velocity is None:
            ax_velocity = ax_altitude.twinx()

        ax_altitude = self.plot_altitude_vs_time(ax=ax_altitude)

        ax_velocity.plot([], [])
        ax_velocity = self.plot_total_velocity_vs_time(ax=ax_velocity)

        ax_altitude.get_legend().remove()
        ax_velocity.get_legend().remove()

        ax_altitude.figure.legend(
            loc='upper right',
            bbox_to_anchor=(1, 1),
            bbox_transform=ax_altitude.transAxes,
        )

        return ax_altitude, ax_velocity

    def copy(self) -> 'Trajectory':
        return Trajectory(
            time_start=self.time_start.copy(),
            time_mission=self.time_mission.copy(),
            altitude=self.altitude.copy(),
            latitude=self.latitude.copy(),
            longitude=self.longitude.copy(),
            velocity=self.velocity.copy(),

        )

    def view(self) -> 'Trajectory':
        return Trajectory(
            time_start=self.time_start,
            time_mission=self.time_mission,
            altitude=self.altitude,
            latitude=self.latitude,
            longitude=self.longitude,
            velocity=self.velocity,

        )


@dataclasses.dataclass
class Event(
    mixin.Colorable,
    mixin.Named,
):
    time_mission: typ.Optional[u.Quantity] = None

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            time_start: typ.Optional[astropy.time.Time] = None,
    ) -> plt.Axes:

        if ax is None:
            _, ax = plt.subplots()

        if time_start is not None:
            time = (time_start + self.time_mission).to_datetime()
        else:
            time = self.time_mission

        color = self.color
        if color is None:
            color = next(ax._get_lines.prop_cycler)['color']
        ax.axvline(x=time, color=color, label=self.name, linestyle='--', linewidth=1)

        return ax


@dataclasses.dataclass
class Timeline:
    t0: Event = dataclasses.field(default_factory=lambda: Event(name=Name('t = 0'), time_mission=0 * u.s))
    rail_release: Event = dataclasses.field(default_factory=lambda: Event(name=Name('rail release')))
    terrier_burnout: Event = dataclasses.field(default_factory=lambda: Event(name=Name('Terrier burnout')))
    black_brant_ignition: Event = dataclasses.field(default_factory=lambda: Event(name=Name('Black Brant ignition')))
    canard_decouple: Event = dataclasses.field(default_factory=lambda: Event(name=Name('S-19L canard decouple')))
    black_brant_burnout: Event = dataclasses.field(default_factory=lambda: Event(name=Name('Black Brant burnout')))
    despin: Event = dataclasses.field(default_factory=lambda: Event(name=Name('despin to 0.25 Hz')))
    payload_separation: Event = dataclasses.field(default_factory=lambda: Event(name=Name('payload separation')))
    sparcs_enable: Event = dataclasses.field(default_factory=lambda: Event(name=Name('SPARCS enable')))
    shutter_door_open: Event = dataclasses.field(default_factory=lambda: Event(name=Name('shutter door open')))
    nosecone_eject: Event = dataclasses.field(default_factory=lambda: Event(name=Name('nosecone eject')))
    sparcs_fine_mode_stable: Event = dataclasses.field(
        default_factory=lambda: Event(name=Name('SPARCS fine mode stable')))
    sparcs_rlg_enable: Event = dataclasses.field(default_factory=lambda: Event(name=Name('SPARCS RLG enable')))
    sparcs_rlg_disable: Event = dataclasses.field(default_factory=lambda: Event(name=Name('SPARCS RLG disable')))
    shutter_door_close: Event = dataclasses.field(default_factory=lambda: Event(name=Name('shutter door close')))
    sparcs_spin_up: Event = dataclasses.field(default_factory=lambda: Event(name=Name('SPARCS spin-up')))
    sparcs_vent: Event = dataclasses.field(default_factory=lambda: Event(name=Name('SPARCS vent')))
    ballistic_impact: Event = dataclasses.field(default_factory=lambda: Event(name=Name('ballistic impact')))
    sparcs_disable: Event = dataclasses.field(default_factory=lambda: Event(name=Name('SPARCS disable')))
    parachute_deploy: Event = dataclasses.field(default_factory=lambda: Event(name=Name('parachute deploy')))
    payload_impact: Event = dataclasses.field(default_factory=lambda: Event(name=Name('payload impact on chute')))

    def __iter__(self) -> typ.Iterator[Event]:
        yield self.t0
        yield self.rail_release
        yield self.terrier_burnout
        yield self.black_brant_ignition
        yield self.canard_decouple
        yield self.black_brant_burnout
        yield self.despin
        yield self.payload_separation
        yield self.sparcs_enable
        yield self.shutter_door_open
        yield self.nosecone_eject
        yield self.sparcs_fine_mode_stable
        yield self.sparcs_rlg_enable
        yield self.sparcs_rlg_disable
        yield self.shutter_door_close
        yield self.sparcs_spin_up
        yield self.sparcs_vent
        yield self.ballistic_impact
        yield self.sparcs_disable
        yield self.parachute_deploy
        yield self.payload_impact

    def plot(
            self,
            ax: plt.Axes,
            time_start: typ.Optional[astropy.time.Time] = None,
    ) -> plt.Axes:
        for event in self:
            event.plot(ax=ax, time_start=time_start)
        return ax
