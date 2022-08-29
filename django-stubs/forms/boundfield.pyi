from typing import Any, Dict, Iterable, Iterator, List, Optional, Union, overload

from django.forms.fields import Field
from django.forms.forms import BaseForm
from django.forms.renderers import BaseRenderer
from django.forms.utils import ErrorList
from django.forms.widgets import Widget
from django.utils.functional import _StrOrPromise
from django.utils.safestring import SafeString

_AttrsT = Dict[str, Union[str, bool]]

class BoundField:
    form: BaseForm = ...
    field: Field = ...
    name: str = ...
    html_name: str = ...
    html_initial_name: str = ...
    html_initial_id: str = ...
    label: str = ...
    help_text: _StrOrPromise = ...
    def __init__(self, form: BaseForm, field: Field, name: str) -> None: ...
    @property
    def subwidgets(self) -> List[BoundWidget]: ...
    def __bool__(self) -> bool: ...
    def __iter__(self) -> Iterator[BoundWidget]: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, idx: Union[int, str]) -> BoundWidget: ...
    @overload
    def __getitem__(self, idx: slice) -> List[BoundWidget]: ...
    @property
    def errors(self) -> ErrorList: ...
    def as_widget(
        self, widget: Optional[Widget] = ..., attrs: Optional[_AttrsT] = ..., only_initial: bool = ...
    ) -> SafeString: ...
    def as_text(self, attrs: Optional[_AttrsT] = ..., **kwargs: Any) -> SafeString: ...
    def as_textarea(self, attrs: Optional[_AttrsT] = ..., **kwargs: Any) -> SafeString: ...
    def as_hidden(self, attrs: Optional[_AttrsT] = ..., **kwargs: Any) -> SafeString: ...
    @property
    def data(self) -> Any: ...
    def value(self) -> Any: ...
    def label_tag(
        self, contents: Optional[str] = ..., attrs: Optional[_AttrsT] = ..., label_suffix: Optional[str] = ...
    ) -> SafeString: ...
    def css_classes(self, extra_classes: Union[str, Iterable[str], None] = ...) -> str: ...
    @property
    def is_hidden(self) -> bool: ...
    @property
    def auto_id(self) -> str: ...
    @property
    def id_for_label(self) -> str: ...
    @property
    def initial(self) -> Any: ...
    def build_widget_attrs(self, attrs: _AttrsT, widget: Optional[Widget] = ...) -> _AttrsT: ...
    @property
    def widget_type(self) -> str: ...

class BoundWidget:
    parent_widget: Widget = ...
    data: Dict[str, Any] = ...
    renderer: BaseRenderer = ...
    def __init__(self, parent_widget: Widget, data: Dict[str, Any], renderer: BaseRenderer) -> None: ...
    def tag(self, wrap_label: bool = ...) -> SafeString: ...
    @property
    def template_name(self) -> str: ...
    @property
    def id_for_label(self) -> str: ...
    @property
    def choice_label(self) -> str: ...
