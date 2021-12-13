import typing as typ
import abc
import dataclasses
import astropy.units as u
from ezdxf.addons.r12writer import R12FastStreamWriter
import kgpy.transform


WritableMixinT = typ.TypeVar('WritableMixinT', bound='WritableMixin')


@dataclasses.dataclass
class WritableMixin(abc.ABC):

    @abc.abstractmethod
    def write_to_dxf(
            self: WritableMixinT,
            file_writer: R12FastStreamWriter,
            unit: u.Unit,
            transform_extra: typ.Optional[kgpy.transform.rigid.Transform] = None,
    ) -> None:
        pass
