from typing import Optional, Union

from mypy.checker import TypeChecker, fill_typevars
from mypy.nodes import (
    GDEF,
    CallExpr,
    Decorator,
    FuncBase,
    FuncDef,
    MemberExpr,
    NameExpr,
    OverloadedFuncDef,
    RefExpr,
    StrExpr,
    SymbolTableNode,
    TypeInfo,
    Var,
)
from mypy.plugin import AttributeContext, DynamicClassDefContext, MethodContext
from mypy.semanal import SemanticAnalyzer
from mypy.types import AnyType, CallableType, Instance, ProperType
from mypy.types import Type as MypyType
from mypy.types import TypeOfAny, TypeType
from typing_extensions import Final

from mypy_django_plugin import errorcodes
from mypy_django_plugin.lib import fullnames, helpers

MANAGER_METHODS_RETURNING_QUERYSET: Final = frozenset(
    (
        "alias",
        "all",
        "annotate",
        "complex_filter",
        "defer",
        "difference",
        "distinct",
        "exclude",
        "extra",
        "filter",
        "intersection",
        "none",
        "only",
        "order_by",
        "prefetch_related",
        "reverse",
        "select_for_update",
        "select_related",
        "union",
        "using",
    )
)


def get_method_type_from_dynamic_manager(
    api: TypeChecker, method_name: str, manager_instance: Instance
) -> Optional[ProperType]:
    """
    Attempt to resolve a method on a manager that was built from '.from_queryset'
    """

    manager_type_info = manager_instance.type

    if (
        "django" not in manager_type_info.metadata
        or "from_queryset_manager" not in manager_type_info.metadata["django"]
    ):
        # Manager isn't dynamically added
        return None

    queryset_fullname = manager_type_info.metadata["django"]["from_queryset_manager"]
    assert isinstance(queryset_fullname, str)
    queryset_info = helpers.lookup_fully_qualified_typeinfo(api, queryset_fullname)
    assert queryset_info is not None

    def get_funcdef_type(definition: Union[FuncBase, Decorator, None]) -> Optional[ProperType]:
        # TODO: Handle @overload?
        if isinstance(definition, FuncBase) and not isinstance(definition, OverloadedFuncDef):
            return definition.type
        elif isinstance(definition, Decorator):
            return definition.func.type
        return None

    method_type = get_funcdef_type(queryset_info.get_method(method_name))
    if method_type is None:
        return None

    assert isinstance(method_type, CallableType)

    variables = method_type.variables
    ret_type = method_type.ret_type

    # For methods on the manager that return a queryset we need to override the
    # return type to be the actual queryset class, not the base QuerySet that's
    # used by the typing stubs.
    if method_name in MANAGER_METHODS_RETURNING_QUERYSET:
        ret_type = Instance(queryset_info, manager_instance.args)
        variables = []

    # Drop any 'self' argument as our manager is already initialized
    return method_type.copy_modified(
        arg_types=method_type.arg_types[1:],
        arg_kinds=method_type.arg_kinds[1:],
        arg_names=method_type.arg_names[1:],
        variables=variables,
        ret_type=ret_type,
    )


def get_method_type_from_reverse_manager(
    api: TypeChecker, method_name: str, manager_type_info: TypeInfo
) -> Optional[ProperType]:
    """
    Attempts to resolve a reverse manager's method via the '_default_manager' manager on the related model
    From Django docs:
      "By default the RelatedManager used for reverse relations is a subclass of the default manager for that model."
    Ref: https://docs.djangoproject.com/en/dev/topics/db/queries/#using-a-custom-reverse-manager
    """
    is_reverse_manager = (
        "django" in manager_type_info.metadata and "related_manager_to_model" in manager_type_info.metadata["django"]
    )
    if not is_reverse_manager:
        return None

    related_model_fullname = manager_type_info.metadata["django"]["related_manager_to_model"]
    assert isinstance(related_model_fullname, str)
    model_info = helpers.lookup_fully_qualified_typeinfo(api, related_model_fullname)
    if model_info is None:
        return None

    # We should _always_ have a '_default_manager' on a model
    assert "_default_manager" in model_info.names
    assert isinstance(model_info.names["_default_manager"].node, Var)
    manager_instance = model_info.names["_default_manager"].node.type
    return (
        get_method_type_from_dynamic_manager(api, method_name, manager_instance)
        # TODO: Can we assert on None and Instance?
        if manager_instance is not None and isinstance(manager_instance, Instance)
        else None
    )


def resolve_manager_method_from_instance(instance: Instance, method_name: str, ctx: AttributeContext) -> MypyType:

    api = helpers.get_typechecker_api(ctx)
    method_type = get_method_type_from_dynamic_manager(
        api, method_name, instance
    ) or get_method_type_from_reverse_manager(api, method_name, instance.type)

    return method_type if method_type is not None else ctx.default_attr_type


def resolve_manager_method(ctx: AttributeContext) -> MypyType:
    """
    A 'get_attribute_hook' that is intended to be invoked whenever the TypeChecker encounters
    an attribute on a class that has 'django.db.models.BaseManager' as a base.
    """
    # Skip (method) type that is currently something other than Any
    if not isinstance(ctx.default_attr_type, AnyType):
        return ctx.default_attr_type

    # (Current state is:) We wouldn't end up here when looking up a method from a custom _manager_.
    # That's why we only attempt to lookup the method for either a dynamically added or reverse manager.
    if isinstance(ctx.context, MemberExpr):
        method_name = ctx.context.name
    elif isinstance(ctx.context, CallExpr) and isinstance(ctx.context.callee, MemberExpr):
        method_name = ctx.context.callee.name
    else:
        ctx.api.fail("Unable to resolve return type of queryset/manager method", ctx.context)
        return AnyType(TypeOfAny.from_error)

    if isinstance(ctx.type, Instance):
        return resolve_manager_method_from_instance(instance=ctx.type, method_name=method_name, ctx=ctx)
    else:
        ctx.api.fail(f'Unable to resolve return type of queryset/manager method "{method_name}"', ctx.context)
        return AnyType(TypeOfAny.from_error)


def merge_queryset_into_manager(
    api: SemanticAnalyzer,
    manager_info: TypeInfo,
    queryset_info: TypeInfo,
    manager_base: TypeInfo,
    class_name: str,
) -> Optional[TypeInfo]:
    """
    Merges a queryset definition with a manager definition.
    As a reference one can look at the 3-arg call definition for ``builtins.type`` to
    get an idea of what kind of class we're trying to express a type for.
    """
    # Stash the queryset fullname so that our 'resolve_manager_method' attribute hook
    # can fetch the method from that QuerySet class
    manager_info.metadata["django"] = {"from_queryset_manager": queryset_info.fullname}

    manager_base.metadata.setdefault("from_queryset_managers", {})
    # The `__module__` value of the manager type created by Django's `.from_queryset`
    # is `django.db.models.manager`. But `basic_new_typeinfo` defaults to what is
    # currently being processed, so we'll map that together through metadata.
    manager_base.metadata["from_queryset_managers"][f"django.db.models.manager.{class_name}"] = manager_info.fullname

    # So that the plugin will reparameterize the manager when it is constructed inside of a Model definition
    helpers.add_new_manager_base(api, manager_info.fullname)

    self_type = fill_typevars(manager_info)
    assert isinstance(self_type, Instance)

    # We collect and mark up all methods before django.db.models.query.QuerySet as class members
    for class_mro_info in queryset_info.mro:
        if class_mro_info.fullname == fullnames.QUERYSET_CLASS_FULLNAME:
            break
        for name, sym in class_mro_info.names.items():
            if not isinstance(sym.node, (FuncDef, Decorator)):
                continue
            # Insert the queryset method name as a class member. Note that the type of
            # the method is set as Any. Figuring out the type is the job of the
            # 'resolve_manager_method' attribute hook, which comes later.
            #
            # class BaseManagerFromMyQuerySet(BaseManager):
            #    queryset_method: Any = ...
            #
            helpers.add_new_sym_for_info(
                manager_info,
                name=name,
                sym_type=AnyType(TypeOfAny.special_form),
            )

    # For methods on BaseManager that return a queryset we need to update the
    # return type to be the actual queryset subclass used. This is done by
    # adding the methods as attributes with type Any to the manager class,
    # similar to how custom queryset methods are handled above. The actual type
    # of these methods are resolved in resolve_manager_method.
    for name in MANAGER_METHODS_RETURNING_QUERYSET:
        helpers.add_new_sym_for_info(
            manager_info,
            name=name,
            sym_type=AnyType(TypeOfAny.special_form),
        )


def create_new_manager_class_from_from_queryset_method(ctx: DynamicClassDefContext) -> None:
    """
    Insert a new manager class node for a: '<Name> = <Manager>.from_queryset(<QuerySet>)'.
    When the assignment expression lives at module level.
    """
    semanal_api = helpers.get_semanal_api(ctx)

    # Don't redeclare the manager class if we've already defined it.
    manager_sym = semanal_api.lookup_current_scope(ctx.name)
    if manager_sym and isinstance(manager_sym.node, TypeInfo):
        # This is just a deferral run where our work is already finished
        return

    callee = ctx.call.callee
    assert isinstance(callee, MemberExpr)
    assert isinstance(callee.expr, RefExpr)

    manager_base = callee.expr.node
    if manager_base is None:
        if not semanal_api.final_iteration:
            semanal_api.defer()
        return

    assert isinstance(manager_base, TypeInfo)

    passed_queryset = ctx.call.args[0]
    assert isinstance(passed_queryset, NameExpr)

    if passed_queryset.fullname is None:
        # In some cases, due to the way the semantic analyzer works, only passed_queryset.name is available.
        # But it should be analyzed again, so this isn't a problem.
        return
    elif semanal_api.is_class_scope():
        # We force `.from_queryset` to be called _outside_ of a class body. So we'll
        # skip doing any work if we're inside of one..
        return

    queryset_sym = semanal_api.lookup_fully_qualified_or_none(passed_queryset.fullname)
    assert queryset_sym is not None
    if queryset_sym.node is None:
        if not semanal_api.final_iteration:
            semanal_api.defer()
        return

    queryset_info = queryset_sym.node
    assert isinstance(queryset_info, TypeInfo)

    if len(ctx.call.args) > 1:
        expr = ctx.call.args[1]
        assert isinstance(expr, StrExpr)
        manager_class_name = expr.value
    else:
        manager_class_name = manager_base.name + "From" + queryset_info.name

    manager_base_instance = fill_typevars(manager_base)
    assert isinstance(manager_base_instance, Instance)
    # Create a new `TypeInfo` instance for the manager type
    new_manager_info = semanal_api.basic_new_typeinfo(
        name=manager_class_name, basetype_or_fallback=manager_base_instance, line=ctx.call.line
    )
    new_manager_info.type_vars = manager_base.type_vars
    new_manager_info.line = ctx.call.line
    new_manager_info.defn.type_vars = manager_base.defn.type_vars
    new_manager_info.defn.line = ctx.call.line
    new_manager_info.metaclass_type = new_manager_info.calculate_metaclass_type()

    try:
        merge_queryset_into_manager(
            api=semanal_api,
            manager_info=new_manager_info,
            queryset_info=queryset_info,
            manager_base=manager_base,
            class_name=manager_class_name,
        )
    except helpers.IncompleteDefnException:
        if not semanal_api.final_iteration:
            semanal_api.defer()
        return

    symbol_kind = semanal_api.current_symbol_kind()
    # Annotate the module variable as `<Variable>: Type[<NewManager[Any]>]` as the model
    # type won't be defined on variable level.
    var = Var(
        name=ctx.name,
        type=TypeType(Instance(new_manager_info, [AnyType(TypeOfAny.from_omitted_generics)])),
    )
    var.info = new_manager_info
    var._fullname = f"{semanal_api.cur_mod_id}.{ctx.name}"
    var.is_inferred = True
    # Note: Order of `add_symbol_table_node` calls matter. Case being if
    # `ctx.name == new_manager_info.name`, then we'd _only_ like the type and not the
    # `Var` to exist..
    assert semanal_api.add_symbol_table_node(ctx.name, SymbolTableNode(symbol_kind, var, plugin_generated=True))
    # Insert the new manager dynamic class
    assert semanal_api.add_symbol_table_node(
        new_manager_info.name, SymbolTableNode(symbol_kind, new_manager_info, plugin_generated=True)
    )


def create_new_manager_class_from_as_manager_method(ctx: DynamicClassDefContext) -> None:
    """
    Insert a new manager class node for a

    ```
    <manager name> = <QuerySet>.as_manager()
    ```
    """
    semanal_api = helpers.get_semanal_api(ctx)
    # Don't redeclare the manager class if we've already defined it.
    manager_node = semanal_api.lookup_current_scope(ctx.name)
    if manager_node and manager_node.type is not None:
        # This is just a deferral run where our work is already finished
        return

    manager_sym = semanal_api.lookup_fully_qualified_or_none(fullnames.MANAGER_CLASS_FULLNAME)
    assert manager_sym is not None
    manager_base = manager_sym.node
    if manager_base is None:
        if not semanal_api.final_iteration:
            semanal_api.defer()
        return

    assert isinstance(manager_base, TypeInfo)

    callee = ctx.call.callee
    assert isinstance(callee, MemberExpr)
    assert isinstance(callee.expr, RefExpr)

    queryset_info = callee.expr.node
    if queryset_info is None:
        if not semanal_api.final_iteration:
            semanal_api.defer()
        return

    assert isinstance(queryset_info, TypeInfo)

    manager_class_name = manager_base.name + "From" + queryset_info.name
    current_module = semanal_api.modules[semanal_api.cur_mod_id]
    existing_sym = current_module.names.get(manager_class_name)
    if (
        existing_sym is not None
        and isinstance(existing_sym.node, TypeInfo)
        and existing_sym.node.metadata.get("django", {}).get("from_queryset_manager") == queryset_info.fullname
    ):
        # Reuse an identical, already generated, manager
        new_manager_info = existing_sym.node
    else:
        manager_base_instance = fill_typevars(manager_base)
        assert isinstance(manager_base_instance, Instance)
        new_manager_info = helpers.add_new_class_for_module(
            module=current_module,
            name=manager_class_name,
            bases=[manager_base_instance],
        )
        new_manager_info.type_vars = manager_base.type_vars
        new_manager_info.line = ctx.call.line
        new_manager_info.defn.type_vars = manager_base.defn.type_vars
        new_manager_info.defn.line = ctx.call.line

        try:
            merge_queryset_into_manager(
                api=semanal_api,
                manager_info=new_manager_info,
                queryset_info=queryset_info,
                manager_base=manager_base,
                class_name=manager_class_name,
            )
        except helpers.IncompleteDefnException:
            if not semanal_api.final_iteration:
                semanal_api.defer()
            return

    # Whenever `<QuerySet>.as_manager()` isn't called at class level, we want to ensure
    # that the variable is an instance of our generated manager. Instead of the return
    # value of `.as_manager()`. Though model argument is populated as `Any`.
    # `transformers.models.AddManagers` will populate a model's manager(s), when it
    # finds it on class level.
    var = Var(name=ctx.name, type=Instance(new_manager_info, [AnyType(TypeOfAny.from_omitted_generics)]))
    var.info = new_manager_info
    var._fullname = f"{current_module.fullname}.{ctx.name}"
    var.is_inferred = True
    # Note: Order of `add_symbol_table_node` calls matters. Depending on what level
    # we've found the `.as_manager()` call. Point here being that we want to replace the
    # `.as_manager` return value with our newly created manager.
    assert semanal_api.add_symbol_table_node(
        ctx.name, SymbolTableNode(semanal_api.current_symbol_kind(), var, plugin_generated=True)
    )
    assert semanal_api.add_symbol_table_node(
        # We'll use `new_manager_info.name` instead of `manager_class_name` here
        # to handle possible name collisions, as it's unique.
        new_manager_info.name,
        # Note that the generated manager type is always inserted at module level
        SymbolTableNode(GDEF, new_manager_info, plugin_generated=True),
    )


def fail_if_manager_type_created_in_model_body(ctx: MethodContext) -> MypyType:
    """
    Method hook that checks if method `<Manager>.from_queryset` is called inside a model class body.

    Doing so won't, for instance, trigger the dynamic class hook(`create_new_manager_class_from_from_queryset_method`)
    for managers.
    """
    api = helpers.get_typechecker_api(ctx)
    outer_model_info = api.scope.active_class()
    if not outer_model_info or not outer_model_info.has_base(fullnames.MODEL_CLASS_FULLNAME):
        # Not inside a model class definition
        return ctx.default_return_type

    api.fail("`.from_queryset` called from inside model class body", ctx.context, code=errorcodes.MANAGER_UNTYPED)
    return ctx.default_return_type
