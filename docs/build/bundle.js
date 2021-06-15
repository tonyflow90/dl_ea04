
(function(l, r) { if (l.getElementById('livereloadscript')) return; r = l.createElement('script'); r.async = 1; r.src = '//' + (window.location.host || 'localhost').split(':')[0] + ':35729/livereload.js?snipver=1'; r.id = 'livereloadscript'; l.getElementsByTagName('head')[0].appendChild(r) })(window.document);
var app = (function () {
    'use strict';

    function noop() { }
    const identity = x => x;
    function assign(tar, src) {
        // @ts-ignore
        for (const k in src)
            tar[k] = src[k];
        return tar;
    }
    function add_location(element, file, line, column, char) {
        element.__svelte_meta = {
            loc: { file, line, column, char }
        };
    }
    function run(fn) {
        return fn();
    }
    function blank_object() {
        return Object.create(null);
    }
    function run_all(fns) {
        fns.forEach(run);
    }
    function is_function(thing) {
        return typeof thing === 'function';
    }
    function safe_not_equal(a, b) {
        return a != a ? b == b : a !== b || ((a && typeof a === 'object') || typeof a === 'function');
    }
    function is_empty(obj) {
        return Object.keys(obj).length === 0;
    }
    function create_slot(definition, ctx, $$scope, fn) {
        if (definition) {
            const slot_ctx = get_slot_context(definition, ctx, $$scope, fn);
            return definition[0](slot_ctx);
        }
    }
    function get_slot_context(definition, ctx, $$scope, fn) {
        return definition[1] && fn
            ? assign($$scope.ctx.slice(), definition[1](fn(ctx)))
            : $$scope.ctx;
    }
    function get_slot_changes(definition, $$scope, dirty, fn) {
        if (definition[2] && fn) {
            const lets = definition[2](fn(dirty));
            if ($$scope.dirty === undefined) {
                return lets;
            }
            if (typeof lets === 'object') {
                const merged = [];
                const len = Math.max($$scope.dirty.length, lets.length);
                for (let i = 0; i < len; i += 1) {
                    merged[i] = $$scope.dirty[i] | lets[i];
                }
                return merged;
            }
            return $$scope.dirty | lets;
        }
        return $$scope.dirty;
    }
    function update_slot(slot, slot_definition, ctx, $$scope, dirty, get_slot_changes_fn, get_slot_context_fn) {
        const slot_changes = get_slot_changes(slot_definition, $$scope, dirty, get_slot_changes_fn);
        if (slot_changes) {
            const slot_context = get_slot_context(slot_definition, ctx, $$scope, get_slot_context_fn);
            slot.p(slot_context, slot_changes);
        }
    }
    function exclude_internal_props(props) {
        const result = {};
        for (const k in props)
            if (k[0] !== '$')
                result[k] = props[k];
        return result;
    }
    function null_to_empty(value) {
        return value == null ? '' : value;
    }
    function action_destroyer(action_result) {
        return action_result && is_function(action_result.destroy) ? action_result.destroy : noop;
    }

    const is_client = typeof window !== 'undefined';
    let now = is_client
        ? () => window.performance.now()
        : () => Date.now();
    let raf = is_client ? cb => requestAnimationFrame(cb) : noop;

    const tasks = new Set();
    function run_tasks(now) {
        tasks.forEach(task => {
            if (!task.c(now)) {
                tasks.delete(task);
                task.f();
            }
        });
        if (tasks.size !== 0)
            raf(run_tasks);
    }
    /**
     * Creates a new task that runs on each raf frame
     * until it returns a falsy value or is aborted
     */
    function loop(callback) {
        let task;
        if (tasks.size === 0)
            raf(run_tasks);
        return {
            promise: new Promise(fulfill => {
                tasks.add(task = { c: callback, f: fulfill });
            }),
            abort() {
                tasks.delete(task);
            }
        };
    }

    function append(target, node) {
        target.appendChild(node);
    }
    function insert(target, node, anchor) {
        target.insertBefore(node, anchor || null);
    }
    function detach(node) {
        node.parentNode.removeChild(node);
    }
    function destroy_each(iterations, detaching) {
        for (let i = 0; i < iterations.length; i += 1) {
            if (iterations[i])
                iterations[i].d(detaching);
        }
    }
    function element(name) {
        return document.createElement(name);
    }
    function svg_element(name) {
        return document.createElementNS('http://www.w3.org/2000/svg', name);
    }
    function text(data) {
        return document.createTextNode(data);
    }
    function space() {
        return text(' ');
    }
    function empty() {
        return text('');
    }
    function listen(node, event, handler, options) {
        node.addEventListener(event, handler, options);
        return () => node.removeEventListener(event, handler, options);
    }
    function attr(node, attribute, value) {
        if (value == null)
            node.removeAttribute(attribute);
        else if (node.getAttribute(attribute) !== value)
            node.setAttribute(attribute, value);
    }
    function set_attributes(node, attributes) {
        // @ts-ignore
        const descriptors = Object.getOwnPropertyDescriptors(node.__proto__);
        for (const key in attributes) {
            if (attributes[key] == null) {
                node.removeAttribute(key);
            }
            else if (key === 'style') {
                node.style.cssText = attributes[key];
            }
            else if (key === '__value') {
                node.value = node[key] = attributes[key];
            }
            else if (descriptors[key] && descriptors[key].set) {
                node[key] = attributes[key];
            }
            else {
                attr(node, key, attributes[key]);
            }
        }
    }
    function set_custom_element_data(node, prop, value) {
        if (prop in node) {
            node[prop] = typeof node[prop] === 'boolean' && value === '' ? true : value;
        }
        else {
            attr(node, prop, value);
        }
    }
    function to_number(value) {
        return value === '' ? null : +value;
    }
    function children(element) {
        return Array.from(element.childNodes);
    }
    function set_input_value(input, value) {
        input.value = value == null ? '' : value;
    }
    function set_style(node, key, value, important) {
        node.style.setProperty(key, value, important ? 'important' : '');
    }
    function toggle_class(element, name, toggle) {
        element.classList[toggle ? 'add' : 'remove'](name);
    }
    function custom_event(type, detail) {
        const e = document.createEvent('CustomEvent');
        e.initCustomEvent(type, false, false, detail);
        return e;
    }
    class HtmlTag {
        constructor(anchor = null) {
            this.a = anchor;
            this.e = this.n = null;
        }
        m(html, target, anchor = null) {
            if (!this.e) {
                this.e = element(target.nodeName);
                this.t = target;
                this.h(html);
            }
            this.i(anchor);
        }
        h(html) {
            this.e.innerHTML = html;
            this.n = Array.from(this.e.childNodes);
        }
        i(anchor) {
            for (let i = 0; i < this.n.length; i += 1) {
                insert(this.t, this.n[i], anchor);
            }
        }
        p(html) {
            this.d();
            this.h(html);
            this.i(this.a);
        }
        d() {
            this.n.forEach(detach);
        }
    }

    const active_docs = new Set();
    let active = 0;
    // https://github.com/darkskyapp/string-hash/blob/master/index.js
    function hash(str) {
        let hash = 5381;
        let i = str.length;
        while (i--)
            hash = ((hash << 5) - hash) ^ str.charCodeAt(i);
        return hash >>> 0;
    }
    function create_rule(node, a, b, duration, delay, ease, fn, uid = 0) {
        const step = 16.666 / duration;
        let keyframes = '{\n';
        for (let p = 0; p <= 1; p += step) {
            const t = a + (b - a) * ease(p);
            keyframes += p * 100 + `%{${fn(t, 1 - t)}}\n`;
        }
        const rule = keyframes + `100% {${fn(b, 1 - b)}}\n}`;
        const name = `__svelte_${hash(rule)}_${uid}`;
        const doc = node.ownerDocument;
        active_docs.add(doc);
        const stylesheet = doc.__svelte_stylesheet || (doc.__svelte_stylesheet = doc.head.appendChild(element('style')).sheet);
        const current_rules = doc.__svelte_rules || (doc.__svelte_rules = {});
        if (!current_rules[name]) {
            current_rules[name] = true;
            stylesheet.insertRule(`@keyframes ${name} ${rule}`, stylesheet.cssRules.length);
        }
        const animation = node.style.animation || '';
        node.style.animation = `${animation ? `${animation}, ` : ''}${name} ${duration}ms linear ${delay}ms 1 both`;
        active += 1;
        return name;
    }
    function delete_rule(node, name) {
        const previous = (node.style.animation || '').split(', ');
        const next = previous.filter(name
            ? anim => anim.indexOf(name) < 0 // remove specific animation
            : anim => anim.indexOf('__svelte') === -1 // remove all Svelte animations
        );
        const deleted = previous.length - next.length;
        if (deleted) {
            node.style.animation = next.join(', ');
            active -= deleted;
            if (!active)
                clear_rules();
        }
    }
    function clear_rules() {
        raf(() => {
            if (active)
                return;
            active_docs.forEach(doc => {
                const stylesheet = doc.__svelte_stylesheet;
                let i = stylesheet.cssRules.length;
                while (i--)
                    stylesheet.deleteRule(i);
                doc.__svelte_rules = {};
            });
            active_docs.clear();
        });
    }

    let current_component;
    function set_current_component(component) {
        current_component = component;
    }
    function get_current_component() {
        if (!current_component)
            throw new Error('Function called outside component initialization');
        return current_component;
    }
    function onMount(fn) {
        get_current_component().$$.on_mount.push(fn);
    }
    function createEventDispatcher() {
        const component = get_current_component();
        return (type, detail) => {
            const callbacks = component.$$.callbacks[type];
            if (callbacks) {
                // TODO are there situations where events could be dispatched
                // in a server (non-DOM) environment?
                const event = custom_event(type, detail);
                callbacks.slice().forEach(fn => {
                    fn.call(component, event);
                });
            }
        };
    }
    // TODO figure out if we still want to support
    // shorthand events, or if we want to implement
    // a real bubbling mechanism
    function bubble(component, event) {
        const callbacks = component.$$.callbacks[event.type];
        if (callbacks) {
            callbacks.slice().forEach(fn => fn(event));
        }
    }

    const dirty_components = [];
    const binding_callbacks = [];
    const render_callbacks = [];
    const flush_callbacks = [];
    const resolved_promise = Promise.resolve();
    let update_scheduled = false;
    function schedule_update() {
        if (!update_scheduled) {
            update_scheduled = true;
            resolved_promise.then(flush);
        }
    }
    function add_render_callback(fn) {
        render_callbacks.push(fn);
    }
    function add_flush_callback(fn) {
        flush_callbacks.push(fn);
    }
    let flushing = false;
    const seen_callbacks = new Set();
    function flush() {
        if (flushing)
            return;
        flushing = true;
        do {
            // first, call beforeUpdate functions
            // and update components
            for (let i = 0; i < dirty_components.length; i += 1) {
                const component = dirty_components[i];
                set_current_component(component);
                update(component.$$);
            }
            set_current_component(null);
            dirty_components.length = 0;
            while (binding_callbacks.length)
                binding_callbacks.pop()();
            // then, once components are updated, call
            // afterUpdate functions. This may cause
            // subsequent updates...
            for (let i = 0; i < render_callbacks.length; i += 1) {
                const callback = render_callbacks[i];
                if (!seen_callbacks.has(callback)) {
                    // ...so guard against infinite loops
                    seen_callbacks.add(callback);
                    callback();
                }
            }
            render_callbacks.length = 0;
        } while (dirty_components.length);
        while (flush_callbacks.length) {
            flush_callbacks.pop()();
        }
        update_scheduled = false;
        flushing = false;
        seen_callbacks.clear();
    }
    function update($$) {
        if ($$.fragment !== null) {
            $$.update();
            run_all($$.before_update);
            const dirty = $$.dirty;
            $$.dirty = [-1];
            $$.fragment && $$.fragment.p($$.ctx, dirty);
            $$.after_update.forEach(add_render_callback);
        }
    }

    let promise;
    function wait() {
        if (!promise) {
            promise = Promise.resolve();
            promise.then(() => {
                promise = null;
            });
        }
        return promise;
    }
    function dispatch(node, direction, kind) {
        node.dispatchEvent(custom_event(`${direction ? 'intro' : 'outro'}${kind}`));
    }
    const outroing = new Set();
    let outros;
    function group_outros() {
        outros = {
            r: 0,
            c: [],
            p: outros // parent group
        };
    }
    function check_outros() {
        if (!outros.r) {
            run_all(outros.c);
        }
        outros = outros.p;
    }
    function transition_in(block, local) {
        if (block && block.i) {
            outroing.delete(block);
            block.i(local);
        }
    }
    function transition_out(block, local, detach, callback) {
        if (block && block.o) {
            if (outroing.has(block))
                return;
            outroing.add(block);
            outros.c.push(() => {
                outroing.delete(block);
                if (callback) {
                    if (detach)
                        block.d(1);
                    callback();
                }
            });
            block.o(local);
        }
    }
    const null_transition = { duration: 0 };
    function create_bidirectional_transition(node, fn, params, intro) {
        let config = fn(node, params);
        let t = intro ? 0 : 1;
        let running_program = null;
        let pending_program = null;
        let animation_name = null;
        function clear_animation() {
            if (animation_name)
                delete_rule(node, animation_name);
        }
        function init(program, duration) {
            const d = program.b - t;
            duration *= Math.abs(d);
            return {
                a: t,
                b: program.b,
                d,
                duration,
                start: program.start,
                end: program.start + duration,
                group: program.group
            };
        }
        function go(b) {
            const { delay = 0, duration = 300, easing = identity, tick = noop, css } = config || null_transition;
            const program = {
                start: now() + delay,
                b
            };
            if (!b) {
                // @ts-ignore todo: improve typings
                program.group = outros;
                outros.r += 1;
            }
            if (running_program || pending_program) {
                pending_program = program;
            }
            else {
                // if this is an intro, and there's a delay, we need to do
                // an initial tick and/or apply CSS animation immediately
                if (css) {
                    clear_animation();
                    animation_name = create_rule(node, t, b, duration, delay, easing, css);
                }
                if (b)
                    tick(0, 1);
                running_program = init(program, duration);
                add_render_callback(() => dispatch(node, b, 'start'));
                loop(now => {
                    if (pending_program && now > pending_program.start) {
                        running_program = init(pending_program, duration);
                        pending_program = null;
                        dispatch(node, running_program.b, 'start');
                        if (css) {
                            clear_animation();
                            animation_name = create_rule(node, t, running_program.b, running_program.duration, 0, easing, config.css);
                        }
                    }
                    if (running_program) {
                        if (now >= running_program.end) {
                            tick(t = running_program.b, 1 - t);
                            dispatch(node, running_program.b, 'end');
                            if (!pending_program) {
                                // we're done
                                if (running_program.b) {
                                    // intro — we can tidy up immediately
                                    clear_animation();
                                }
                                else {
                                    // outro — needs to be coordinated
                                    if (!--running_program.group.r)
                                        run_all(running_program.group.c);
                                }
                            }
                            running_program = null;
                        }
                        else if (now >= running_program.start) {
                            const p = now - running_program.start;
                            t = running_program.a + running_program.d * easing(p / running_program.duration);
                            tick(t, 1 - t);
                        }
                    }
                    return !!(running_program || pending_program);
                });
            }
        }
        return {
            run(b) {
                if (is_function(config)) {
                    wait().then(() => {
                        // @ts-ignore
                        config = config();
                        go(b);
                    });
                }
                else {
                    go(b);
                }
            },
            end() {
                clear_animation();
                running_program = pending_program = null;
            }
        };
    }

    const globals = (typeof window !== 'undefined'
        ? window
        : typeof globalThis !== 'undefined'
            ? globalThis
            : global);

    function get_spread_update(levels, updates) {
        const update = {};
        const to_null_out = {};
        const accounted_for = { $$scope: 1 };
        let i = levels.length;
        while (i--) {
            const o = levels[i];
            const n = updates[i];
            if (n) {
                for (const key in o) {
                    if (!(key in n))
                        to_null_out[key] = 1;
                }
                for (const key in n) {
                    if (!accounted_for[key]) {
                        update[key] = n[key];
                        accounted_for[key] = 1;
                    }
                }
                levels[i] = n;
            }
            else {
                for (const key in o) {
                    accounted_for[key] = 1;
                }
            }
        }
        for (const key in to_null_out) {
            if (!(key in update))
                update[key] = undefined;
        }
        return update;
    }
    function get_spread_object(spread_props) {
        return typeof spread_props === 'object' && spread_props !== null ? spread_props : {};
    }

    function bind(component, name, callback) {
        const index = component.$$.props[name];
        if (index !== undefined) {
            component.$$.bound[index] = callback;
            callback(component.$$.ctx[index]);
        }
    }
    function create_component(block) {
        block && block.c();
    }
    function mount_component(component, target, anchor, customElement) {
        const { fragment, on_mount, on_destroy, after_update } = component.$$;
        fragment && fragment.m(target, anchor);
        if (!customElement) {
            // onMount happens before the initial afterUpdate
            add_render_callback(() => {
                const new_on_destroy = on_mount.map(run).filter(is_function);
                if (on_destroy) {
                    on_destroy.push(...new_on_destroy);
                }
                else {
                    // Edge case - component was destroyed immediately,
                    // most likely as a result of a binding initialising
                    run_all(new_on_destroy);
                }
                component.$$.on_mount = [];
            });
        }
        after_update.forEach(add_render_callback);
    }
    function destroy_component(component, detaching) {
        const $$ = component.$$;
        if ($$.fragment !== null) {
            run_all($$.on_destroy);
            $$.fragment && $$.fragment.d(detaching);
            // TODO null out other refs, including component.$$ (but need to
            // preserve final state?)
            $$.on_destroy = $$.fragment = null;
            $$.ctx = [];
        }
    }
    function make_dirty(component, i) {
        if (component.$$.dirty[0] === -1) {
            dirty_components.push(component);
            schedule_update();
            component.$$.dirty.fill(0);
        }
        component.$$.dirty[(i / 31) | 0] |= (1 << (i % 31));
    }
    function init(component, options, instance, create_fragment, not_equal, props, dirty = [-1]) {
        const parent_component = current_component;
        set_current_component(component);
        const $$ = component.$$ = {
            fragment: null,
            ctx: null,
            // state
            props,
            update: noop,
            not_equal,
            bound: blank_object(),
            // lifecycle
            on_mount: [],
            on_destroy: [],
            on_disconnect: [],
            before_update: [],
            after_update: [],
            context: new Map(parent_component ? parent_component.$$.context : options.context || []),
            // everything else
            callbacks: blank_object(),
            dirty,
            skip_bound: false
        };
        let ready = false;
        $$.ctx = instance
            ? instance(component, options.props || {}, (i, ret, ...rest) => {
                const value = rest.length ? rest[0] : ret;
                if ($$.ctx && not_equal($$.ctx[i], $$.ctx[i] = value)) {
                    if (!$$.skip_bound && $$.bound[i])
                        $$.bound[i](value);
                    if (ready)
                        make_dirty(component, i);
                }
                return ret;
            })
            : [];
        $$.update();
        ready = true;
        run_all($$.before_update);
        // `false` as a special case of no DOM component
        $$.fragment = create_fragment ? create_fragment($$.ctx) : false;
        if (options.target) {
            if (options.hydrate) {
                const nodes = children(options.target);
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                $$.fragment && $$.fragment.l(nodes);
                nodes.forEach(detach);
            }
            else {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                $$.fragment && $$.fragment.c();
            }
            if (options.intro)
                transition_in(component.$$.fragment);
            mount_component(component, options.target, options.anchor, options.customElement);
            flush();
        }
        set_current_component(parent_component);
    }
    /**
     * Base class for Svelte components. Used when dev=false.
     */
    class SvelteComponent {
        $destroy() {
            destroy_component(this, 1);
            this.$destroy = noop;
        }
        $on(type, callback) {
            const callbacks = (this.$$.callbacks[type] || (this.$$.callbacks[type] = []));
            callbacks.push(callback);
            return () => {
                const index = callbacks.indexOf(callback);
                if (index !== -1)
                    callbacks.splice(index, 1);
            };
        }
        $set($$props) {
            if (this.$$set && !is_empty($$props)) {
                this.$$.skip_bound = true;
                this.$$set($$props);
                this.$$.skip_bound = false;
            }
        }
    }

    function dispatch_dev(type, detail) {
        document.dispatchEvent(custom_event(type, Object.assign({ version: '3.38.2' }, detail)));
    }
    function append_dev(target, node) {
        dispatch_dev('SvelteDOMInsert', { target, node });
        append(target, node);
    }
    function insert_dev(target, node, anchor) {
        dispatch_dev('SvelteDOMInsert', { target, node, anchor });
        insert(target, node, anchor);
    }
    function detach_dev(node) {
        dispatch_dev('SvelteDOMRemove', { node });
        detach(node);
    }
    function listen_dev(node, event, handler, options, has_prevent_default, has_stop_propagation) {
        const modifiers = options === true ? ['capture'] : options ? Array.from(Object.keys(options)) : [];
        if (has_prevent_default)
            modifiers.push('preventDefault');
        if (has_stop_propagation)
            modifiers.push('stopPropagation');
        dispatch_dev('SvelteDOMAddEventListener', { node, event, handler, modifiers });
        const dispose = listen(node, event, handler, options);
        return () => {
            dispatch_dev('SvelteDOMRemoveEventListener', { node, event, handler, modifiers });
            dispose();
        };
    }
    function attr_dev(node, attribute, value) {
        attr(node, attribute, value);
        if (value == null)
            dispatch_dev('SvelteDOMRemoveAttribute', { node, attribute });
        else
            dispatch_dev('SvelteDOMSetAttribute', { node, attribute, value });
    }
    function prop_dev(node, property, value) {
        node[property] = value;
        dispatch_dev('SvelteDOMSetProperty', { node, property, value });
    }
    function set_data_dev(text, data) {
        data = '' + data;
        if (text.wholeText === data)
            return;
        dispatch_dev('SvelteDOMSetData', { node: text, data });
        text.data = data;
    }
    function validate_each_argument(arg) {
        if (typeof arg !== 'string' && !(arg && typeof arg === 'object' && 'length' in arg)) {
            let msg = '{#each} only iterates over array-like objects.';
            if (typeof Symbol === 'function' && arg && Symbol.iterator in arg) {
                msg += ' You can use a spread to convert this iterable into an array.';
            }
            throw new Error(msg);
        }
    }
    function validate_slots(name, slot, keys) {
        for (const slot_key of Object.keys(slot)) {
            if (!~keys.indexOf(slot_key)) {
                console.warn(`<${name}> received an unexpected slot "${slot_key}".`);
            }
        }
    }
    /**
     * Base class for Svelte components with some minor dev-enhancements. Used when dev=true.
     */
    class SvelteComponentDev extends SvelteComponent {
        constructor(options) {
            if (!options || (!options.target && !options.$$inline)) {
                throw new Error("'target' is a required option");
            }
            super();
        }
        $destroy() {
            super.$destroy();
            this.$destroy = () => {
                console.warn('Component was already destroyed'); // eslint-disable-line no-console
            };
        }
        $capture_state() { }
        $inject_state() { }
    }

    /* node_modules\smelte\src\components\Icon\Icon.svelte generated by Svelte v3.38.2 */

    const file$d = "node_modules\\smelte\\src\\components\\Icon\\Icon.svelte";

    function create_fragment$e(ctx) {
    	let i;
    	let i_class_value;
    	let i_style_value;
    	let current;
    	let mounted;
    	let dispose;
    	const default_slot_template = /*#slots*/ ctx[7].default;
    	const default_slot = create_slot(default_slot_template, ctx, /*$$scope*/ ctx[6], null);

    	const block = {
    		c: function create() {
    			i = element("i");
    			if (default_slot) default_slot.c();
    			attr_dev(i, "aria-hidden", "true");
    			attr_dev(i, "class", i_class_value = "material-icons icon text-xl select-none " + /*$$props*/ ctx[5].class + " duration-200 ease-in" + " svelte-zzky5a");
    			attr_dev(i, "style", i_style_value = /*color*/ ctx[4] ? `color: ${/*color*/ ctx[4]}` : "");
    			toggle_class(i, "reverse", /*reverse*/ ctx[2]);
    			toggle_class(i, "tip", /*tip*/ ctx[3]);
    			toggle_class(i, "text-base", /*small*/ ctx[0]);
    			toggle_class(i, "text-xs", /*xs*/ ctx[1]);
    			add_location(i, file$d, 20, 0, 273);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, i, anchor);

    			if (default_slot) {
    				default_slot.m(i, null);
    			}

    			current = true;

    			if (!mounted) {
    				dispose = listen_dev(i, "click", /*click_handler*/ ctx[8], false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, [dirty]) {
    			if (default_slot) {
    				if (default_slot.p && (!current || dirty & /*$$scope*/ 64)) {
    					update_slot(default_slot, default_slot_template, ctx, /*$$scope*/ ctx[6], dirty, null, null);
    				}
    			}

    			if (!current || dirty & /*$$props*/ 32 && i_class_value !== (i_class_value = "material-icons icon text-xl select-none " + /*$$props*/ ctx[5].class + " duration-200 ease-in" + " svelte-zzky5a")) {
    				attr_dev(i, "class", i_class_value);
    			}

    			if (!current || dirty & /*color*/ 16 && i_style_value !== (i_style_value = /*color*/ ctx[4] ? `color: ${/*color*/ ctx[4]}` : "")) {
    				attr_dev(i, "style", i_style_value);
    			}

    			if (dirty & /*$$props, reverse*/ 36) {
    				toggle_class(i, "reverse", /*reverse*/ ctx[2]);
    			}

    			if (dirty & /*$$props, tip*/ 40) {
    				toggle_class(i, "tip", /*tip*/ ctx[3]);
    			}

    			if (dirty & /*$$props, small*/ 33) {
    				toggle_class(i, "text-base", /*small*/ ctx[0]);
    			}

    			if (dirty & /*$$props, xs*/ 34) {
    				toggle_class(i, "text-xs", /*xs*/ ctx[1]);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(default_slot, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(default_slot, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(i);
    			if (default_slot) default_slot.d(detaching);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$e.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$e($$self, $$props, $$invalidate) {
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("Icon", slots, ['default']);
    	let { small = false } = $$props;
    	let { xs = false } = $$props;
    	let { reverse = false } = $$props;
    	let { tip = false } = $$props;
    	let { color = "default" } = $$props;

    	function click_handler(event) {
    		bubble($$self, event);
    	}

    	$$self.$$set = $$new_props => {
    		$$invalidate(5, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("small" in $$new_props) $$invalidate(0, small = $$new_props.small);
    		if ("xs" in $$new_props) $$invalidate(1, xs = $$new_props.xs);
    		if ("reverse" in $$new_props) $$invalidate(2, reverse = $$new_props.reverse);
    		if ("tip" in $$new_props) $$invalidate(3, tip = $$new_props.tip);
    		if ("color" in $$new_props) $$invalidate(4, color = $$new_props.color);
    		if ("$$scope" in $$new_props) $$invalidate(6, $$scope = $$new_props.$$scope);
    	};

    	$$self.$capture_state = () => ({ small, xs, reverse, tip, color });

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(5, $$props = assign(assign({}, $$props), $$new_props));
    		if ("small" in $$props) $$invalidate(0, small = $$new_props.small);
    		if ("xs" in $$props) $$invalidate(1, xs = $$new_props.xs);
    		if ("reverse" in $$props) $$invalidate(2, reverse = $$new_props.reverse);
    		if ("tip" in $$props) $$invalidate(3, tip = $$new_props.tip);
    		if ("color" in $$props) $$invalidate(4, color = $$new_props.color);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$props = exclude_internal_props($$props);
    	return [small, xs, reverse, tip, color, $$props, $$scope, slots, click_handler];
    }

    class Icon extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(this, options, instance$e, create_fragment$e, safe_not_equal, {
    			small: 0,
    			xs: 1,
    			reverse: 2,
    			tip: 3,
    			color: 4
    		});

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Icon",
    			options,
    			id: create_fragment$e.name
    		});
    	}

    	get small() {
    		throw new Error("<Icon>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set small(value) {
    		throw new Error("<Icon>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get xs() {
    		throw new Error("<Icon>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set xs(value) {
    		throw new Error("<Icon>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get reverse() {
    		throw new Error("<Icon>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set reverse(value) {
    		throw new Error("<Icon>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get tip() {
    		throw new Error("<Icon>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set tip(value) {
    		throw new Error("<Icon>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get color() {
    		throw new Error("<Icon>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set color(value) {
    		throw new Error("<Icon>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    const noDepth = ["white", "black", "transparent"];

    function getClass(prop, color, depth, defaultDepth) {
      if (noDepth.includes(color)) {
        return `${prop}-${color}`;
      }
      return `${prop}-${color}-${depth || defaultDepth} `;
    }

    function utils(color, defaultDepth = 500) {
      return {
        bg: depth => getClass("bg", color, depth, defaultDepth),
        border: depth => getClass("border", color, depth, defaultDepth),
        txt: depth => getClass("text", color, depth, defaultDepth),
        caret: depth => getClass("caret", color, depth, defaultDepth)
      };
    }

    class ClassBuilder {
      constructor(classes, defaultClasses) {
        this.defaults =
          (typeof classes === "function" ? classes(defaultClasses) : classes) ||
          defaultClasses;

        this.classes = this.defaults;
      }

      flush() {
        this.classes = this.defaults;

        return this;
      }

      extend(...fns) {
        return this;
      }

      get() {
        return this.classes;
      }

      replace(classes, cond = true) {
        if (cond && classes) {
          this.classes = Object.keys(classes).reduce(
            (acc, from) => acc.replace(new RegExp(from, "g"), classes[from]),
            this.classes
          );
        }

        return this;
      }

      remove(classes, cond = true) {
        if (cond && classes) {
          this.classes = classes
            .split(" ")
            .reduce(
              (acc, cur) => acc.replace(new RegExp(cur, "g"), ""),
              this.classes
            );
        }

        return this;
      }

      add(className, cond = true, defaultValue) {
        if (!cond || !className) return this;

        switch (typeof className) {
          case "string":
          default:
            this.classes += ` ${className} `;
            return this;
          case "function":
            this.classes += ` ${className(defaultValue || this.classes)} `;
            return this;
        }
      }
    }

    const defaultReserved = ["class", "add", "remove", "replace", "value"];

    function filterProps(reserved, props) {
      const r = [...reserved, ...defaultReserved];

      return Object.keys(props).reduce(
        (acc, cur) =>
          cur.includes("$$") || cur.includes("Class") || r.includes(cur)
            ? acc
            : { ...acc, [cur]: props[cur] },
        {}
      );
    }

    // Thanks Lagden! https://svelte.dev/repl/61d9178d2b9944f2aa2bfe31612ab09f?version=3.6.7
    function ripple(color, centered) {
      return function(event) {
        const target = event.currentTarget;
        const circle = document.createElement("span");
        const d = Math.max(target.clientWidth, target.clientHeight);

        const removeCircle = () => {
          circle.remove();
          circle.removeEventListener("animationend", removeCircle);
        };

        circle.addEventListener("animationend", removeCircle);
        circle.style.width = circle.style.height = `${d}px`;
        const rect = target.getBoundingClientRect();

        if (centered) {
          circle.classList.add(
            "absolute",
            "top-0",
            "left-0",
            "ripple-centered",
            `bg-${color}-transDark`
          );
        } else {
          circle.style.left = `${event.clientX - rect.left - d / 2}px`;
          circle.style.top = `${event.clientY - rect.top - d / 2}px`;

          circle.classList.add("ripple-normal", `bg-${color}-trans`);
        }

        circle.classList.add("ripple");

        target.appendChild(circle);
      };
    }

    function r(color = "primary", centered = false) {
      return function(node) {
        const onMouseDown = ripple(color, centered);
        node.addEventListener("mousedown", onMouseDown);

        return {
          onDestroy: () => node.removeEventListener("mousedown", onMouseDown),
        };
      };
    }

    /* node_modules\smelte\src\components\Button\Button.svelte generated by Svelte v3.38.2 */
    const file$c = "node_modules\\smelte\\src\\components\\Button\\Button.svelte";

    // (153:0) {:else}
    function create_else_block$2(ctx) {
    	let button;
    	let t;
    	let current;
    	let mounted;
    	let dispose;
    	let if_block = /*icon*/ ctx[3] && create_if_block_2$2(ctx);
    	const default_slot_template = /*#slots*/ ctx[34].default;
    	const default_slot = create_slot(default_slot_template, ctx, /*$$scope*/ ctx[43], null);

    	let button_levels = [
    		{ class: /*classes*/ ctx[1] },
    		/*props*/ ctx[9],
    		{ type: /*type*/ ctx[6] },
    		{ disabled: /*disabled*/ ctx[2] }
    	];

    	let button_data = {};

    	for (let i = 0; i < button_levels.length; i += 1) {
    		button_data = assign(button_data, button_levels[i]);
    	}

    	const block_1 = {
    		c: function create() {
    			button = element("button");
    			if (if_block) if_block.c();
    			t = space();
    			if (default_slot) default_slot.c();
    			set_attributes(button, button_data);
    			add_location(button, file$c, 153, 2, 4075);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, button, anchor);
    			if (if_block) if_block.m(button, null);
    			append_dev(button, t);

    			if (default_slot) {
    				default_slot.m(button, null);
    			}

    			current = true;

    			if (!mounted) {
    				dispose = [
    					action_destroyer(/*ripple*/ ctx[8].call(null, button)),
    					listen_dev(button, "click", /*click_handler_3*/ ctx[42], false, false, false),
    					listen_dev(button, "click", /*click_handler_1*/ ctx[38], false, false, false),
    					listen_dev(button, "mouseover", /*mouseover_handler_1*/ ctx[39], false, false, false),
    					listen_dev(button, "*", /*_handler_1*/ ctx[40], false, false, false)
    				];

    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (/*icon*/ ctx[3]) {
    				if (if_block) {
    					if_block.p(ctx, dirty);

    					if (dirty[0] & /*icon*/ 8) {
    						transition_in(if_block, 1);
    					}
    				} else {
    					if_block = create_if_block_2$2(ctx);
    					if_block.c();
    					transition_in(if_block, 1);
    					if_block.m(button, t);
    				}
    			} else if (if_block) {
    				group_outros();

    				transition_out(if_block, 1, 1, () => {
    					if_block = null;
    				});

    				check_outros();
    			}

    			if (default_slot) {
    				if (default_slot.p && (!current || dirty[1] & /*$$scope*/ 4096)) {
    					update_slot(default_slot, default_slot_template, ctx, /*$$scope*/ ctx[43], dirty, null, null);
    				}
    			}

    			set_attributes(button, button_data = get_spread_update(button_levels, [
    				(!current || dirty[0] & /*classes*/ 2) && { class: /*classes*/ ctx[1] },
    				/*props*/ ctx[9],
    				(!current || dirty[0] & /*type*/ 64) && { type: /*type*/ ctx[6] },
    				(!current || dirty[0] & /*disabled*/ 4) && { disabled: /*disabled*/ ctx[2] }
    			]));
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(if_block);
    			transition_in(default_slot, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(if_block);
    			transition_out(default_slot, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(button);
    			if (if_block) if_block.d();
    			if (default_slot) default_slot.d(detaching);
    			mounted = false;
    			run_all(dispose);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block: block_1,
    		id: create_else_block$2.name,
    		type: "else",
    		source: "(153:0) {:else}",
    		ctx
    	});

    	return block_1;
    }

    // (131:0) {#if href}
    function create_if_block$5(ctx) {
    	let a;
    	let button;
    	let t;
    	let current;
    	let mounted;
    	let dispose;
    	let if_block = /*icon*/ ctx[3] && create_if_block_1$3(ctx);
    	const default_slot_template = /*#slots*/ ctx[34].default;
    	const default_slot = create_slot(default_slot_template, ctx, /*$$scope*/ ctx[43], null);

    	let button_levels = [
    		{ class: /*classes*/ ctx[1] },
    		/*props*/ ctx[9],
    		{ type: /*type*/ ctx[6] },
    		{ disabled: /*disabled*/ ctx[2] }
    	];

    	let button_data = {};

    	for (let i = 0; i < button_levels.length; i += 1) {
    		button_data = assign(button_data, button_levels[i]);
    	}

    	let a_levels = [{ href: /*href*/ ctx[5] }, /*props*/ ctx[9]];
    	let a_data = {};

    	for (let i = 0; i < a_levels.length; i += 1) {
    		a_data = assign(a_data, a_levels[i]);
    	}

    	const block_1 = {
    		c: function create() {
    			a = element("a");
    			button = element("button");
    			if (if_block) if_block.c();
    			t = space();
    			if (default_slot) default_slot.c();
    			set_attributes(button, button_data);
    			add_location(button, file$c, 135, 4, 3762);
    			set_attributes(a, a_data);
    			add_location(a, file$c, 131, 2, 3725);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, a, anchor);
    			append_dev(a, button);
    			if (if_block) if_block.m(button, null);
    			append_dev(button, t);

    			if (default_slot) {
    				default_slot.m(button, null);
    			}

    			current = true;

    			if (!mounted) {
    				dispose = [
    					action_destroyer(/*ripple*/ ctx[8].call(null, button)),
    					listen_dev(button, "click", /*click_handler_2*/ ctx[41], false, false, false),
    					listen_dev(button, "click", /*click_handler*/ ctx[35], false, false, false),
    					listen_dev(button, "mouseover", /*mouseover_handler*/ ctx[36], false, false, false),
    					listen_dev(button, "*", /*_handler*/ ctx[37], false, false, false)
    				];

    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (/*icon*/ ctx[3]) {
    				if (if_block) {
    					if_block.p(ctx, dirty);

    					if (dirty[0] & /*icon*/ 8) {
    						transition_in(if_block, 1);
    					}
    				} else {
    					if_block = create_if_block_1$3(ctx);
    					if_block.c();
    					transition_in(if_block, 1);
    					if_block.m(button, t);
    				}
    			} else if (if_block) {
    				group_outros();

    				transition_out(if_block, 1, 1, () => {
    					if_block = null;
    				});

    				check_outros();
    			}

    			if (default_slot) {
    				if (default_slot.p && (!current || dirty[1] & /*$$scope*/ 4096)) {
    					update_slot(default_slot, default_slot_template, ctx, /*$$scope*/ ctx[43], dirty, null, null);
    				}
    			}

    			set_attributes(button, button_data = get_spread_update(button_levels, [
    				(!current || dirty[0] & /*classes*/ 2) && { class: /*classes*/ ctx[1] },
    				/*props*/ ctx[9],
    				(!current || dirty[0] & /*type*/ 64) && { type: /*type*/ ctx[6] },
    				(!current || dirty[0] & /*disabled*/ 4) && { disabled: /*disabled*/ ctx[2] }
    			]));

    			set_attributes(a, a_data = get_spread_update(a_levels, [
    				(!current || dirty[0] & /*href*/ 32) && { href: /*href*/ ctx[5] },
    				/*props*/ ctx[9]
    			]));
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(if_block);
    			transition_in(default_slot, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(if_block);
    			transition_out(default_slot, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(a);
    			if (if_block) if_block.d();
    			if (default_slot) default_slot.d(detaching);
    			mounted = false;
    			run_all(dispose);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block: block_1,
    		id: create_if_block$5.name,
    		type: "if",
    		source: "(131:0) {#if href}",
    		ctx
    	});

    	return block_1;
    }

    // (165:4) {#if icon}
    function create_if_block_2$2(ctx) {
    	let icon_1;
    	let current;

    	icon_1 = new Icon({
    			props: {
    				class: /*iClasses*/ ctx[7],
    				small: /*small*/ ctx[4],
    				$$slots: { default: [create_default_slot_1$2] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	const block_1 = {
    		c: function create() {
    			create_component(icon_1.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(icon_1, target, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const icon_1_changes = {};
    			if (dirty[0] & /*iClasses*/ 128) icon_1_changes.class = /*iClasses*/ ctx[7];
    			if (dirty[0] & /*small*/ 16) icon_1_changes.small = /*small*/ ctx[4];

    			if (dirty[0] & /*icon*/ 8 | dirty[1] & /*$$scope*/ 4096) {
    				icon_1_changes.$$scope = { dirty, ctx };
    			}

    			icon_1.$set(icon_1_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(icon_1.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(icon_1.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(icon_1, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block: block_1,
    		id: create_if_block_2$2.name,
    		type: "if",
    		source: "(165:4) {#if icon}",
    		ctx
    	});

    	return block_1;
    }

    // (166:6) <Icon class={iClasses} {small}>
    function create_default_slot_1$2(ctx) {
    	let t;

    	const block_1 = {
    		c: function create() {
    			t = text(/*icon*/ ctx[3]);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, t, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*icon*/ 8) set_data_dev(t, /*icon*/ ctx[3]);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block: block_1,
    		id: create_default_slot_1$2.name,
    		type: "slot",
    		source: "(166:6) <Icon class={iClasses} {small}>",
    		ctx
    	});

    	return block_1;
    }

    // (147:6) {#if icon}
    function create_if_block_1$3(ctx) {
    	let icon_1;
    	let current;

    	icon_1 = new Icon({
    			props: {
    				class: /*iClasses*/ ctx[7],
    				small: /*small*/ ctx[4],
    				$$slots: { default: [create_default_slot$4] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	const block_1 = {
    		c: function create() {
    			create_component(icon_1.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(icon_1, target, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const icon_1_changes = {};
    			if (dirty[0] & /*iClasses*/ 128) icon_1_changes.class = /*iClasses*/ ctx[7];
    			if (dirty[0] & /*small*/ 16) icon_1_changes.small = /*small*/ ctx[4];

    			if (dirty[0] & /*icon*/ 8 | dirty[1] & /*$$scope*/ 4096) {
    				icon_1_changes.$$scope = { dirty, ctx };
    			}

    			icon_1.$set(icon_1_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(icon_1.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(icon_1.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(icon_1, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block: block_1,
    		id: create_if_block_1$3.name,
    		type: "if",
    		source: "(147:6) {#if icon}",
    		ctx
    	});

    	return block_1;
    }

    // (148:8) <Icon class={iClasses} {small}>
    function create_default_slot$4(ctx) {
    	let t;

    	const block_1 = {
    		c: function create() {
    			t = text(/*icon*/ ctx[3]);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, t, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*icon*/ 8) set_data_dev(t, /*icon*/ ctx[3]);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block: block_1,
    		id: create_default_slot$4.name,
    		type: "slot",
    		source: "(148:8) <Icon class={iClasses} {small}>",
    		ctx
    	});

    	return block_1;
    }

    function create_fragment$d(ctx) {
    	let current_block_type_index;
    	let if_block;
    	let if_block_anchor;
    	let current;
    	const if_block_creators = [create_if_block$5, create_else_block$2];
    	const if_blocks = [];

    	function select_block_type(ctx, dirty) {
    		if (/*href*/ ctx[5]) return 0;
    		return 1;
    	}

    	current_block_type_index = select_block_type(ctx);
    	if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);

    	const block_1 = {
    		c: function create() {
    			if_block.c();
    			if_block_anchor = empty();
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			if_blocks[current_block_type_index].m(target, anchor);
    			insert_dev(target, if_block_anchor, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			let previous_block_index = current_block_type_index;
    			current_block_type_index = select_block_type(ctx);

    			if (current_block_type_index === previous_block_index) {
    				if_blocks[current_block_type_index].p(ctx, dirty);
    			} else {
    				group_outros();

    				transition_out(if_blocks[previous_block_index], 1, 1, () => {
    					if_blocks[previous_block_index] = null;
    				});

    				check_outros();
    				if_block = if_blocks[current_block_type_index];

    				if (!if_block) {
    					if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
    					if_block.c();
    				} else {
    					if_block.p(ctx, dirty);
    				}

    				transition_in(if_block, 1);
    				if_block.m(if_block_anchor.parentNode, if_block_anchor);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(if_block);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(if_block);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if_blocks[current_block_type_index].d(detaching);
    			if (detaching) detach_dev(if_block_anchor);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block: block_1,
    		id: create_fragment$d.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block_1;
    }

    const classesDefault$4 = "z-10 py-2 px-4 uppercase text-sm font-medium relative overflow-hidden";
    const basicDefault = "text-white duration-200 ease-in";
    const outlinedDefault = "bg-transparent border border-solid";
    const textDefault = "bg-transparent border-none px-4 hover:bg-transparent";
    const iconDefault = "p-4 flex items-center select-none";
    const fabDefault = "hover:bg-transparent";
    const smallDefault = "pt-1 pb-1 pl-2 pr-2 text-xs";
    const disabledDefault = "bg-gray-300 text-gray-500 dark:bg-dark-400 pointer-events-none hover:bg-gray-300 cursor-default";
    const elevationDefault = "hover:shadow shadow";

    function instance$d($$self, $$props, $$invalidate) {
    	let normal;
    	let lighter;
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("Button", slots, ['default']);
    	let { value = false } = $$props;
    	let { outlined = false } = $$props;
    	let { text = false } = $$props;
    	let { block = false } = $$props;
    	let { disabled = false } = $$props;
    	let { icon = null } = $$props;
    	let { small = false } = $$props;
    	let { light = false } = $$props;
    	let { dark = false } = $$props;
    	let { flat = false } = $$props;
    	let { iconClass = "" } = $$props;
    	let { color = "primary" } = $$props;
    	let { href = null } = $$props;
    	let { fab = false } = $$props;
    	let { type = "button" } = $$props;
    	let { remove = "" } = $$props;
    	let { add = "" } = $$props;
    	let { replace = {} } = $$props;
    	let { classes = classesDefault$4 } = $$props;
    	let { basicClasses = basicDefault } = $$props;
    	let { outlinedClasses = outlinedDefault } = $$props;
    	let { textClasses = textDefault } = $$props;
    	let { iconClasses = iconDefault } = $$props;
    	let { fabClasses = fabDefault } = $$props;
    	let { smallClasses = smallDefault } = $$props;
    	let { disabledClasses = disabledDefault } = $$props;
    	let { elevationClasses = elevationDefault } = $$props;
    	fab = fab || text && icon;
    	const basic = !outlined && !text && !fab;
    	const elevation = (basic || icon) && !disabled && !flat && !text;
    	let Classes = i => i;
    	let iClasses = i => i;
    	let shade = 0;
    	const { bg, border, txt } = utils(color);
    	const cb = new ClassBuilder(classes, classesDefault$4);
    	let iconCb;

    	if (icon) {
    		iconCb = new ClassBuilder(iconClass);
    	}

    	const ripple = r(text || fab || outlined ? color : "white");

    	const props = filterProps(
    		[
    			"outlined",
    			"text",
    			"color",
    			"block",
    			"disabled",
    			"icon",
    			"small",
    			"light",
    			"dark",
    			"flat",
    			"add",
    			"remove",
    			"replace"
    		],
    		$$props
    	);

    	function click_handler(event) {
    		bubble($$self, event);
    	}

    	function mouseover_handler(event) {
    		bubble($$self, event);
    	}

    	function _handler(event) {
    		bubble($$self, event);
    	}

    	function click_handler_1(event) {
    		bubble($$self, event);
    	}

    	function mouseover_handler_1(event) {
    		bubble($$self, event);
    	}

    	function _handler_1(event) {
    		bubble($$self, event);
    	}

    	const click_handler_2 = () => $$invalidate(0, value = !value);
    	const click_handler_3 = () => $$invalidate(0, value = !value);

    	$$self.$$set = $$new_props => {
    		$$invalidate(51, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("value" in $$new_props) $$invalidate(0, value = $$new_props.value);
    		if ("outlined" in $$new_props) $$invalidate(11, outlined = $$new_props.outlined);
    		if ("text" in $$new_props) $$invalidate(12, text = $$new_props.text);
    		if ("block" in $$new_props) $$invalidate(13, block = $$new_props.block);
    		if ("disabled" in $$new_props) $$invalidate(2, disabled = $$new_props.disabled);
    		if ("icon" in $$new_props) $$invalidate(3, icon = $$new_props.icon);
    		if ("small" in $$new_props) $$invalidate(4, small = $$new_props.small);
    		if ("light" in $$new_props) $$invalidate(14, light = $$new_props.light);
    		if ("dark" in $$new_props) $$invalidate(15, dark = $$new_props.dark);
    		if ("flat" in $$new_props) $$invalidate(16, flat = $$new_props.flat);
    		if ("iconClass" in $$new_props) $$invalidate(17, iconClass = $$new_props.iconClass);
    		if ("color" in $$new_props) $$invalidate(18, color = $$new_props.color);
    		if ("href" in $$new_props) $$invalidate(5, href = $$new_props.href);
    		if ("fab" in $$new_props) $$invalidate(10, fab = $$new_props.fab);
    		if ("type" in $$new_props) $$invalidate(6, type = $$new_props.type);
    		if ("remove" in $$new_props) $$invalidate(19, remove = $$new_props.remove);
    		if ("add" in $$new_props) $$invalidate(20, add = $$new_props.add);
    		if ("replace" in $$new_props) $$invalidate(21, replace = $$new_props.replace);
    		if ("classes" in $$new_props) $$invalidate(1, classes = $$new_props.classes);
    		if ("basicClasses" in $$new_props) $$invalidate(22, basicClasses = $$new_props.basicClasses);
    		if ("outlinedClasses" in $$new_props) $$invalidate(23, outlinedClasses = $$new_props.outlinedClasses);
    		if ("textClasses" in $$new_props) $$invalidate(24, textClasses = $$new_props.textClasses);
    		if ("iconClasses" in $$new_props) $$invalidate(25, iconClasses = $$new_props.iconClasses);
    		if ("fabClasses" in $$new_props) $$invalidate(26, fabClasses = $$new_props.fabClasses);
    		if ("smallClasses" in $$new_props) $$invalidate(27, smallClasses = $$new_props.smallClasses);
    		if ("disabledClasses" in $$new_props) $$invalidate(28, disabledClasses = $$new_props.disabledClasses);
    		if ("elevationClasses" in $$new_props) $$invalidate(29, elevationClasses = $$new_props.elevationClasses);
    		if ("$$scope" in $$new_props) $$invalidate(43, $$scope = $$new_props.$$scope);
    	};

    	$$self.$capture_state = () => ({
    		Icon,
    		utils,
    		ClassBuilder,
    		filterProps,
    		createRipple: r,
    		value,
    		outlined,
    		text,
    		block,
    		disabled,
    		icon,
    		small,
    		light,
    		dark,
    		flat,
    		iconClass,
    		color,
    		href,
    		fab,
    		type,
    		remove,
    		add,
    		replace,
    		classesDefault: classesDefault$4,
    		basicDefault,
    		outlinedDefault,
    		textDefault,
    		iconDefault,
    		fabDefault,
    		smallDefault,
    		disabledDefault,
    		elevationDefault,
    		classes,
    		basicClasses,
    		outlinedClasses,
    		textClasses,
    		iconClasses,
    		fabClasses,
    		smallClasses,
    		disabledClasses,
    		elevationClasses,
    		basic,
    		elevation,
    		Classes,
    		iClasses,
    		shade,
    		bg,
    		border,
    		txt,
    		cb,
    		iconCb,
    		ripple,
    		props,
    		normal,
    		lighter
    	});

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(51, $$props = assign(assign({}, $$props), $$new_props));
    		if ("value" in $$props) $$invalidate(0, value = $$new_props.value);
    		if ("outlined" in $$props) $$invalidate(11, outlined = $$new_props.outlined);
    		if ("text" in $$props) $$invalidate(12, text = $$new_props.text);
    		if ("block" in $$props) $$invalidate(13, block = $$new_props.block);
    		if ("disabled" in $$props) $$invalidate(2, disabled = $$new_props.disabled);
    		if ("icon" in $$props) $$invalidate(3, icon = $$new_props.icon);
    		if ("small" in $$props) $$invalidate(4, small = $$new_props.small);
    		if ("light" in $$props) $$invalidate(14, light = $$new_props.light);
    		if ("dark" in $$props) $$invalidate(15, dark = $$new_props.dark);
    		if ("flat" in $$props) $$invalidate(16, flat = $$new_props.flat);
    		if ("iconClass" in $$props) $$invalidate(17, iconClass = $$new_props.iconClass);
    		if ("color" in $$props) $$invalidate(18, color = $$new_props.color);
    		if ("href" in $$props) $$invalidate(5, href = $$new_props.href);
    		if ("fab" in $$props) $$invalidate(10, fab = $$new_props.fab);
    		if ("type" in $$props) $$invalidate(6, type = $$new_props.type);
    		if ("remove" in $$props) $$invalidate(19, remove = $$new_props.remove);
    		if ("add" in $$props) $$invalidate(20, add = $$new_props.add);
    		if ("replace" in $$props) $$invalidate(21, replace = $$new_props.replace);
    		if ("classes" in $$props) $$invalidate(1, classes = $$new_props.classes);
    		if ("basicClasses" in $$props) $$invalidate(22, basicClasses = $$new_props.basicClasses);
    		if ("outlinedClasses" in $$props) $$invalidate(23, outlinedClasses = $$new_props.outlinedClasses);
    		if ("textClasses" in $$props) $$invalidate(24, textClasses = $$new_props.textClasses);
    		if ("iconClasses" in $$props) $$invalidate(25, iconClasses = $$new_props.iconClasses);
    		if ("fabClasses" in $$props) $$invalidate(26, fabClasses = $$new_props.fabClasses);
    		if ("smallClasses" in $$props) $$invalidate(27, smallClasses = $$new_props.smallClasses);
    		if ("disabledClasses" in $$props) $$invalidate(28, disabledClasses = $$new_props.disabledClasses);
    		if ("elevationClasses" in $$props) $$invalidate(29, elevationClasses = $$new_props.elevationClasses);
    		if ("Classes" in $$props) Classes = $$new_props.Classes;
    		if ("iClasses" in $$props) $$invalidate(7, iClasses = $$new_props.iClasses);
    		if ("shade" in $$props) $$invalidate(30, shade = $$new_props.shade);
    		if ("iconCb" in $$props) $$invalidate(31, iconCb = $$new_props.iconCb);
    		if ("normal" in $$props) $$invalidate(32, normal = $$new_props.normal);
    		if ("lighter" in $$props) $$invalidate(33, lighter = $$new_props.lighter);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		if ($$self.$$.dirty[0] & /*light, dark, shade*/ 1073790976) {
    			{
    				$$invalidate(30, shade = light ? 200 : 0);
    				$$invalidate(30, shade = dark ? -400 : shade);
    			}
    		}

    		if ($$self.$$.dirty[0] & /*shade*/ 1073741824) {
    			$$invalidate(32, normal = 500 - shade);
    		}

    		if ($$self.$$.dirty[0] & /*shade*/ 1073741824) {
    			$$invalidate(33, lighter = 400 - shade);
    		}

    		$$invalidate(1, classes = cb.flush().add(basicClasses, basic, basicDefault).add(`${bg(normal)} hover:${bg(lighter)}`, basic).add(elevationClasses, elevation, elevationDefault).add(outlinedClasses, outlined, outlinedDefault).add(`${border(lighter)} ${txt(normal)} hover:${bg("trans")} dark-hover:${bg("transDark")}`, outlined).add(`${txt(lighter)}`, text).add(textClasses, text, textDefault).add(iconClasses, icon, iconDefault).remove("py-2", icon).remove(txt(lighter), fab).add(disabledClasses, disabled, disabledDefault).add(smallClasses, small, smallDefault).add("flex items-center justify-center h-8 w-8", small && icon).add("border-solid", outlined).add("rounded-full", icon).add("w-full", block).add("rounded", basic || outlined || text).add("button", !icon).add(fabClasses, fab, fabDefault).add(`hover:${bg("transLight")}`, fab).add($$props.class).remove(remove).replace(replace).add(add).get());

    		if ($$self.$$.dirty[0] & /*fab, iconClass*/ 132096 | $$self.$$.dirty[1] & /*iconCb*/ 1) {
    			if (iconCb) {
    				$$invalidate(7, iClasses = iconCb.flush().add(txt(), fab && !iconClass).get());
    			}
    		}
    	};

    	$$props = exclude_internal_props($$props);

    	return [
    		value,
    		classes,
    		disabled,
    		icon,
    		small,
    		href,
    		type,
    		iClasses,
    		ripple,
    		props,
    		fab,
    		outlined,
    		text,
    		block,
    		light,
    		dark,
    		flat,
    		iconClass,
    		color,
    		remove,
    		add,
    		replace,
    		basicClasses,
    		outlinedClasses,
    		textClasses,
    		iconClasses,
    		fabClasses,
    		smallClasses,
    		disabledClasses,
    		elevationClasses,
    		shade,
    		iconCb,
    		normal,
    		lighter,
    		slots,
    		click_handler,
    		mouseover_handler,
    		_handler,
    		click_handler_1,
    		mouseover_handler_1,
    		_handler_1,
    		click_handler_2,
    		click_handler_3,
    		$$scope
    	];
    }

    class Button extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(
    			this,
    			options,
    			instance$d,
    			create_fragment$d,
    			safe_not_equal,
    			{
    				value: 0,
    				outlined: 11,
    				text: 12,
    				block: 13,
    				disabled: 2,
    				icon: 3,
    				small: 4,
    				light: 14,
    				dark: 15,
    				flat: 16,
    				iconClass: 17,
    				color: 18,
    				href: 5,
    				fab: 10,
    				type: 6,
    				remove: 19,
    				add: 20,
    				replace: 21,
    				classes: 1,
    				basicClasses: 22,
    				outlinedClasses: 23,
    				textClasses: 24,
    				iconClasses: 25,
    				fabClasses: 26,
    				smallClasses: 27,
    				disabledClasses: 28,
    				elevationClasses: 29
    			},
    			[-1, -1]
    		);

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Button",
    			options,
    			id: create_fragment$d.name
    		});
    	}

    	get value() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set value(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get outlined() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set outlined(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get text() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set text(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get block() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set block(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get disabled() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set disabled(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get icon() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set icon(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get small() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set small(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get light() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set light(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get dark() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set dark(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get flat() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set flat(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get iconClass() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set iconClass(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get color() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set color(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get href() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set href(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get fab() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set fab(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get type() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set type(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get remove() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set remove(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get add() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set add(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get replace() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set replace(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get classes() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set classes(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get basicClasses() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set basicClasses(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get outlinedClasses() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set outlinedClasses(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get textClasses() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set textClasses(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get iconClasses() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set iconClasses(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get fabClasses() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set fabClasses(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get smallClasses() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set smallClasses(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get disabledClasses() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set disabledClasses(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get elevationClasses() {
    		throw new Error("<Button>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set elevationClasses(value) {
    		throw new Error("<Button>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    function cubicOut(t) {
        const f = t - 1.0;
        return f * f * f + 1.0;
    }
    function quadIn(t) {
        return t * t;
    }
    function quadOut(t) {
        return -t * (t - 2.0);
    }

    function fly(node, { delay = 0, duration = 400, easing = cubicOut, x = 0, y = 0, opacity = 0 } = {}) {
        const style = getComputedStyle(node);
        const target_opacity = +style.opacity;
        const transform = style.transform === 'none' ? '' : style.transform;
        const od = target_opacity * (1 - opacity);
        return {
            delay,
            duration,
            easing,
            css: (t, u) => `
			transform: ${transform} translate(${(1 - t) * x}px, ${(1 - t) * y}px);
			opacity: ${target_opacity - (od * u)}`
        };
    }

    /* node_modules\smelte\src\components\List\ListItem.svelte generated by Svelte v3.38.2 */
    const file$b = "node_modules\\smelte\\src\\components\\List\\ListItem.svelte";

    // (59:2) {#if icon}
    function create_if_block_1$2(ctx) {
    	let icon_1;
    	let current;

    	icon_1 = new Icon({
    			props: {
    				class: "pr-6",
    				small: /*dense*/ ctx[3],
    				$$slots: { default: [create_default_slot$3] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	const block = {
    		c: function create() {
    			create_component(icon_1.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(icon_1, target, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const icon_1_changes = {};
    			if (dirty & /*dense*/ 8) icon_1_changes.small = /*dense*/ ctx[3];

    			if (dirty & /*$$scope, icon*/ 4194305) {
    				icon_1_changes.$$scope = { dirty, ctx };
    			}

    			icon_1.$set(icon_1_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(icon_1.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(icon_1.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(icon_1, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_1$2.name,
    		type: "if",
    		source: "(59:2) {#if icon}",
    		ctx
    	});

    	return block;
    }

    // (60:4) <Icon       class="pr-6"       small={dense}     >
    function create_default_slot$3(ctx) {
    	let t;

    	const block = {
    		c: function create() {
    			t = text(/*icon*/ ctx[0]);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, t, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty & /*icon*/ 1) set_data_dev(t, /*icon*/ ctx[0]);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot$3.name,
    		type: "slot",
    		source: "(60:4) <Icon       class=\\\"pr-6\\\"       small={dense}     >",
    		ctx
    	});

    	return block;
    }

    // (70:12) {text}
    function fallback_block$3(ctx) {
    	let t;

    	const block = {
    		c: function create() {
    			t = text(/*text*/ ctx[1]);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, t, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty & /*text*/ 2) set_data_dev(t, /*text*/ ctx[1]);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: fallback_block$3.name,
    		type: "fallback",
    		source: "(70:12) {text}",
    		ctx
    	});

    	return block;
    }

    // (72:4) {#if subheading}
    function create_if_block$4(ctx) {
    	let div;
    	let t;

    	const block = {
    		c: function create() {
    			div = element("div");
    			t = text(/*subheading*/ ctx[2]);
    			attr_dev(div, "class", /*subheadingClasses*/ ctx[5]);
    			add_location(div, file$b, 72, 6, 1808);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div, anchor);
    			append_dev(div, t);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty & /*subheading*/ 4) set_data_dev(t, /*subheading*/ ctx[2]);

    			if (dirty & /*subheadingClasses*/ 32) {
    				attr_dev(div, "class", /*subheadingClasses*/ ctx[5]);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block$4.name,
    		type: "if",
    		source: "(72:4) {#if subheading}",
    		ctx
    	});

    	return block;
    }

    function create_fragment$c(ctx) {
    	let li;
    	let t0;
    	let div1;
    	let div0;
    	let div0_class_value;
    	let t1;
    	let current;
    	let mounted;
    	let dispose;
    	let if_block0 = /*icon*/ ctx[0] && create_if_block_1$2(ctx);
    	const default_slot_template = /*#slots*/ ctx[20].default;
    	const default_slot = create_slot(default_slot_template, ctx, /*$$scope*/ ctx[22], null);
    	const default_slot_or_fallback = default_slot || fallback_block$3(ctx);
    	let if_block1 = /*subheading*/ ctx[2] && create_if_block$4(ctx);

    	const block = {
    		c: function create() {
    			li = element("li");
    			if (if_block0) if_block0.c();
    			t0 = space();
    			div1 = element("div");
    			div0 = element("div");
    			if (default_slot_or_fallback) default_slot_or_fallback.c();
    			t1 = space();
    			if (if_block1) if_block1.c();
    			attr_dev(div0, "class", div0_class_value = /*$$props*/ ctx[9].class);
    			add_location(div0, file$b, 68, 4, 1716);
    			attr_dev(div1, "class", "flex flex-col p-0");
    			add_location(div1, file$b, 67, 2, 1680);
    			attr_dev(li, "class", /*c*/ ctx[6]);
    			attr_dev(li, "tabindex", /*tabindex*/ ctx[4]);
    			add_location(li, file$b, 51, 0, 1479);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, li, anchor);
    			if (if_block0) if_block0.m(li, null);
    			append_dev(li, t0);
    			append_dev(li, div1);
    			append_dev(div1, div0);

    			if (default_slot_or_fallback) {
    				default_slot_or_fallback.m(div0, null);
    			}

    			append_dev(div1, t1);
    			if (if_block1) if_block1.m(div1, null);
    			current = true;

    			if (!mounted) {
    				dispose = [
    					action_destroyer(/*ripple*/ ctx[7].call(null, li)),
    					listen_dev(li, "keypress", /*change*/ ctx[8], false, false, false),
    					listen_dev(li, "click", /*change*/ ctx[8], false, false, false),
    					listen_dev(li, "click", /*click_handler*/ ctx[21], false, false, false)
    				];

    				mounted = true;
    			}
    		},
    		p: function update(ctx, [dirty]) {
    			if (/*icon*/ ctx[0]) {
    				if (if_block0) {
    					if_block0.p(ctx, dirty);

    					if (dirty & /*icon*/ 1) {
    						transition_in(if_block0, 1);
    					}
    				} else {
    					if_block0 = create_if_block_1$2(ctx);
    					if_block0.c();
    					transition_in(if_block0, 1);
    					if_block0.m(li, t0);
    				}
    			} else if (if_block0) {
    				group_outros();

    				transition_out(if_block0, 1, 1, () => {
    					if_block0 = null;
    				});

    				check_outros();
    			}

    			if (default_slot) {
    				if (default_slot.p && (!current || dirty & /*$$scope*/ 4194304)) {
    					update_slot(default_slot, default_slot_template, ctx, /*$$scope*/ ctx[22], dirty, null, null);
    				}
    			} else {
    				if (default_slot_or_fallback && default_slot_or_fallback.p && dirty & /*text*/ 2) {
    					default_slot_or_fallback.p(ctx, dirty);
    				}
    			}

    			if (!current || dirty & /*$$props*/ 512 && div0_class_value !== (div0_class_value = /*$$props*/ ctx[9].class)) {
    				attr_dev(div0, "class", div0_class_value);
    			}

    			if (/*subheading*/ ctx[2]) {
    				if (if_block1) {
    					if_block1.p(ctx, dirty);
    				} else {
    					if_block1 = create_if_block$4(ctx);
    					if_block1.c();
    					if_block1.m(div1, null);
    				}
    			} else if (if_block1) {
    				if_block1.d(1);
    				if_block1 = null;
    			}

    			if (!current || dirty & /*c*/ 64) {
    				attr_dev(li, "class", /*c*/ ctx[6]);
    			}

    			if (!current || dirty & /*tabindex*/ 16) {
    				attr_dev(li, "tabindex", /*tabindex*/ ctx[4]);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(if_block0);
    			transition_in(default_slot_or_fallback, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(if_block0);
    			transition_out(default_slot_or_fallback, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(li);
    			if (if_block0) if_block0.d();
    			if (default_slot_or_fallback) default_slot_or_fallback.d(detaching);
    			if (if_block1) if_block1.d();
    			mounted = false;
    			run_all(dispose);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$c.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    const classesDefault$3 = "focus:bg-gray-50 dark-focus:bg-gray-700 hover:bg-gray-transDark relative overflow-hidden duration-100 p-4 cursor-pointer text-gray-700 dark:text-gray-100 flex items-center z-10";
    const selectedClassesDefault = "bg-gray-200 dark:bg-primary-transLight";
    const subheadingClassesDefault = "text-gray-600 p-0 text-sm";

    function instance$c($$self, $$props, $$invalidate) {
    	let c;
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("ListItem", slots, ['default']);
    	let { icon = "" } = $$props;
    	let { id = "" } = $$props;
    	let { value = "" } = $$props;
    	let { text = "" } = $$props;
    	let { subheading = "" } = $$props;
    	let { disabled = false } = $$props;
    	let { dense = false } = $$props;
    	let { selected = false } = $$props;
    	let { tabindex = null } = $$props;
    	let { selectedClasses = selectedClassesDefault } = $$props;
    	let { subheadingClasses = subheadingClassesDefault } = $$props;
    	let { to = "" } = $$props;
    	const item = null;
    	const items = [];
    	const level = null;
    	const ripple = r();
    	const dispatch = createEventDispatcher();

    	function change() {
    		if (disabled) return;
    		$$invalidate(10, value = id);
    		dispatch("change", id, to);
    	}

    	let { classes = classesDefault$3 } = $$props;
    	const cb = new ClassBuilder(classes, classesDefault$3);

    	function click_handler(event) {
    		bubble($$self, event);
    	}

    	$$self.$$set = $$new_props => {
    		$$invalidate(9, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("icon" in $$new_props) $$invalidate(0, icon = $$new_props.icon);
    		if ("id" in $$new_props) $$invalidate(11, id = $$new_props.id);
    		if ("value" in $$new_props) $$invalidate(10, value = $$new_props.value);
    		if ("text" in $$new_props) $$invalidate(1, text = $$new_props.text);
    		if ("subheading" in $$new_props) $$invalidate(2, subheading = $$new_props.subheading);
    		if ("disabled" in $$new_props) $$invalidate(12, disabled = $$new_props.disabled);
    		if ("dense" in $$new_props) $$invalidate(3, dense = $$new_props.dense);
    		if ("selected" in $$new_props) $$invalidate(13, selected = $$new_props.selected);
    		if ("tabindex" in $$new_props) $$invalidate(4, tabindex = $$new_props.tabindex);
    		if ("selectedClasses" in $$new_props) $$invalidate(14, selectedClasses = $$new_props.selectedClasses);
    		if ("subheadingClasses" in $$new_props) $$invalidate(5, subheadingClasses = $$new_props.subheadingClasses);
    		if ("to" in $$new_props) $$invalidate(15, to = $$new_props.to);
    		if ("classes" in $$new_props) $$invalidate(19, classes = $$new_props.classes);
    		if ("$$scope" in $$new_props) $$invalidate(22, $$scope = $$new_props.$$scope);
    	};

    	$$self.$capture_state = () => ({
    		ClassBuilder,
    		createEventDispatcher,
    		Icon,
    		createRipple: r,
    		classesDefault: classesDefault$3,
    		selectedClassesDefault,
    		subheadingClassesDefault,
    		icon,
    		id,
    		value,
    		text,
    		subheading,
    		disabled,
    		dense,
    		selected,
    		tabindex,
    		selectedClasses,
    		subheadingClasses,
    		to,
    		item,
    		items,
    		level,
    		ripple,
    		dispatch,
    		change,
    		classes,
    		cb,
    		c
    	});

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(9, $$props = assign(assign({}, $$props), $$new_props));
    		if ("icon" in $$props) $$invalidate(0, icon = $$new_props.icon);
    		if ("id" in $$props) $$invalidate(11, id = $$new_props.id);
    		if ("value" in $$props) $$invalidate(10, value = $$new_props.value);
    		if ("text" in $$props) $$invalidate(1, text = $$new_props.text);
    		if ("subheading" in $$props) $$invalidate(2, subheading = $$new_props.subheading);
    		if ("disabled" in $$props) $$invalidate(12, disabled = $$new_props.disabled);
    		if ("dense" in $$props) $$invalidate(3, dense = $$new_props.dense);
    		if ("selected" in $$props) $$invalidate(13, selected = $$new_props.selected);
    		if ("tabindex" in $$props) $$invalidate(4, tabindex = $$new_props.tabindex);
    		if ("selectedClasses" in $$props) $$invalidate(14, selectedClasses = $$new_props.selectedClasses);
    		if ("subheadingClasses" in $$props) $$invalidate(5, subheadingClasses = $$new_props.subheadingClasses);
    		if ("to" in $$props) $$invalidate(15, to = $$new_props.to);
    		if ("classes" in $$props) $$invalidate(19, classes = $$new_props.classes);
    		if ("c" in $$props) $$invalidate(6, c = $$new_props.c);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		$$invalidate(6, c = cb.flush().add(selectedClasses, selected, selectedClassesDefault).add("py-2", dense).add("text-gray-600", disabled).add($$props.class).get());
    	};

    	$$props = exclude_internal_props($$props);

    	return [
    		icon,
    		text,
    		subheading,
    		dense,
    		tabindex,
    		subheadingClasses,
    		c,
    		ripple,
    		change,
    		$$props,
    		value,
    		id,
    		disabled,
    		selected,
    		selectedClasses,
    		to,
    		item,
    		items,
    		level,
    		classes,
    		slots,
    		click_handler,
    		$$scope
    	];
    }

    class ListItem extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(this, options, instance$c, create_fragment$c, safe_not_equal, {
    			icon: 0,
    			id: 11,
    			value: 10,
    			text: 1,
    			subheading: 2,
    			disabled: 12,
    			dense: 3,
    			selected: 13,
    			tabindex: 4,
    			selectedClasses: 14,
    			subheadingClasses: 5,
    			to: 15,
    			item: 16,
    			items: 17,
    			level: 18,
    			classes: 19
    		});

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "ListItem",
    			options,
    			id: create_fragment$c.name
    		});
    	}

    	get icon() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set icon(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get id() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set id(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get value() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set value(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get text() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set text(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get subheading() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set subheading(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get disabled() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set disabled(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get dense() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set dense(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get selected() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set selected(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get tabindex() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set tabindex(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get selectedClasses() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set selectedClasses(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get subheadingClasses() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set subheadingClasses(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get to() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set to(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get item() {
    		return this.$$.ctx[16];
    	}

    	set item(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get items() {
    		return this.$$.ctx[17];
    	}

    	set items(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get level() {
    		return this.$$.ctx[18];
    	}

    	set level(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get classes() {
    		throw new Error("<ListItem>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set classes(value) {
    		throw new Error("<ListItem>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* node_modules\smelte\src\components\List\List.svelte generated by Svelte v3.38.2 */
    const file$a = "node_modules\\smelte\\src\\components\\List\\List.svelte";

    function get_each_context(ctx, list, i) {
    	const child_ctx = ctx.slice();
    	child_ctx[6] = list[i];
    	child_ctx[22] = i;
    	return child_ctx;
    }

    const get_item_slot_changes_1 = dirty => ({
    	item: dirty & /*items*/ 2,
    	dense: dirty & /*dense*/ 4,
    	value: dirty & /*value*/ 1
    });

    const get_item_slot_context_1 = ctx => ({
    	item: /*item*/ ctx[6],
    	dense: /*dense*/ ctx[2],
    	value: /*value*/ ctx[0]
    });

    const get_item_slot_changes = dirty => ({
    	item: dirty & /*items*/ 2,
    	dense: dirty & /*dense*/ 4,
    	value: dirty & /*value*/ 1
    });

    const get_item_slot_context = ctx => ({
    	item: /*item*/ ctx[6],
    	dense: /*dense*/ ctx[2],
    	value: /*value*/ ctx[0]
    });

    // (55:4) {:else}
    function create_else_block$1(ctx) {
    	let current;
    	const item_slot_template = /*#slots*/ ctx[12].item;
    	const item_slot = create_slot(item_slot_template, ctx, /*$$scope*/ ctx[18], get_item_slot_context_1);
    	const item_slot_or_fallback = item_slot || fallback_block_1$2(ctx);

    	const block = {
    		c: function create() {
    			if (item_slot_or_fallback) item_slot_or_fallback.c();
    		},
    		m: function mount(target, anchor) {
    			if (item_slot_or_fallback) {
    				item_slot_or_fallback.m(target, anchor);
    			}

    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			if (item_slot) {
    				if (item_slot.p && (!current || dirty & /*$$scope, items, dense, value*/ 262151)) {
    					update_slot(item_slot, item_slot_template, ctx, /*$$scope*/ ctx[18], dirty, get_item_slot_changes_1, get_item_slot_context_1);
    				}
    			} else {
    				if (item_slot_or_fallback && item_slot_or_fallback.p && dirty & /*items, value, dense*/ 7) {
    					item_slot_or_fallback.p(ctx, dirty);
    				}
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(item_slot_or_fallback, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(item_slot_or_fallback, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (item_slot_or_fallback) item_slot_or_fallback.d(detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_else_block$1.name,
    		type: "else",
    		source: "(55:4) {:else}",
    		ctx
    	});

    	return block;
    }

    // (47:4) {#if item.to !== undefined}
    function create_if_block$3(ctx) {
    	let current;
    	const item_slot_template = /*#slots*/ ctx[12].item;
    	const item_slot = create_slot(item_slot_template, ctx, /*$$scope*/ ctx[18], get_item_slot_context);
    	const item_slot_or_fallback = item_slot || fallback_block$2(ctx);

    	const block = {
    		c: function create() {
    			if (item_slot_or_fallback) item_slot_or_fallback.c();
    		},
    		m: function mount(target, anchor) {
    			if (item_slot_or_fallback) {
    				item_slot_or_fallback.m(target, anchor);
    			}

    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			if (item_slot) {
    				if (item_slot.p && (!current || dirty & /*$$scope, items, dense, value*/ 262151)) {
    					update_slot(item_slot, item_slot_template, ctx, /*$$scope*/ ctx[18], dirty, get_item_slot_changes, get_item_slot_context);
    				}
    			} else {
    				if (item_slot_or_fallback && item_slot_or_fallback.p && dirty & /*items, dense, value*/ 7) {
    					item_slot_or_fallback.p(ctx, dirty);
    				}
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(item_slot_or_fallback, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(item_slot_or_fallback, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (item_slot_or_fallback) item_slot_or_fallback.d(detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block$3.name,
    		type: "if",
    		source: "(47:4) {#if item.to !== undefined}",
    		ctx
    	});

    	return block;
    }

    // (57:8) <ListItem           bind:value           {selectedClasses}           {itemClasses}           {...item}           tabindex={i + 1}           id={id(item)}           selected={value === id(item)}           {dense}           on:change           on:click>
    function create_default_slot_1$1(ctx) {
    	let t_value = getText(/*item*/ ctx[6]) + "";
    	let t;

    	const block = {
    		c: function create() {
    			t = text(t_value);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, t, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty & /*items*/ 2 && t_value !== (t_value = getText(/*item*/ ctx[6]) + "")) set_data_dev(t, t_value);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot_1$1.name,
    		type: "slot",
    		source: "(57:8) <ListItem           bind:value           {selectedClasses}           {itemClasses}           {...item}           tabindex={i + 1}           id={id(item)}           selected={value === id(item)}           {dense}           on:change           on:click>",
    		ctx
    	});

    	return block;
    }

    // (56:47)          
    function fallback_block_1$2(ctx) {
    	let listitem;
    	let updating_value;
    	let t;
    	let current;

    	const listitem_spread_levels = [
    		{
    			selectedClasses: /*selectedClasses*/ ctx[4]
    		},
    		{ itemClasses: /*itemClasses*/ ctx[5] },
    		/*item*/ ctx[6],
    		{ tabindex: /*i*/ ctx[22] + 1 },
    		{ id: id(/*item*/ ctx[6]) },
    		{
    			selected: /*value*/ ctx[0] === id(/*item*/ ctx[6])
    		},
    		{ dense: /*dense*/ ctx[2] }
    	];

    	function listitem_value_binding_1(value) {
    		/*listitem_value_binding_1*/ ctx[15](value);
    	}

    	let listitem_props = {
    		$$slots: { default: [create_default_slot_1$1] },
    		$$scope: { ctx }
    	};

    	for (let i = 0; i < listitem_spread_levels.length; i += 1) {
    		listitem_props = assign(listitem_props, listitem_spread_levels[i]);
    	}

    	if (/*value*/ ctx[0] !== void 0) {
    		listitem_props.value = /*value*/ ctx[0];
    	}

    	listitem = new ListItem({ props: listitem_props, $$inline: true });
    	binding_callbacks.push(() => bind(listitem, "value", listitem_value_binding_1));
    	listitem.$on("change", /*change_handler_1*/ ctx[16]);
    	listitem.$on("click", /*click_handler*/ ctx[17]);

    	const block = {
    		c: function create() {
    			create_component(listitem.$$.fragment);
    			t = space();
    		},
    		m: function mount(target, anchor) {
    			mount_component(listitem, target, anchor);
    			insert_dev(target, t, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const listitem_changes = (dirty & /*selectedClasses, itemClasses, items, id, value, dense*/ 55)
    			? get_spread_update(listitem_spread_levels, [
    					dirty & /*selectedClasses*/ 16 && {
    						selectedClasses: /*selectedClasses*/ ctx[4]
    					},
    					dirty & /*itemClasses*/ 32 && { itemClasses: /*itemClasses*/ ctx[5] },
    					dirty & /*items*/ 2 && get_spread_object(/*item*/ ctx[6]),
    					listitem_spread_levels[3],
    					dirty & /*id, items*/ 2 && { id: id(/*item*/ ctx[6]) },
    					dirty & /*value, id, items*/ 3 && {
    						selected: /*value*/ ctx[0] === id(/*item*/ ctx[6])
    					},
    					dirty & /*dense*/ 4 && { dense: /*dense*/ ctx[2] }
    				])
    			: {};

    			if (dirty & /*$$scope, items*/ 262146) {
    				listitem_changes.$$scope = { dirty, ctx };
    			}

    			if (!updating_value && dirty & /*value*/ 1) {
    				updating_value = true;
    				listitem_changes.value = /*value*/ ctx[0];
    				add_flush_callback(() => updating_value = false);
    			}

    			listitem.$set(listitem_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(listitem.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(listitem.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(listitem, detaching);
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: fallback_block_1$2.name,
    		type: "fallback",
    		source: "(56:47)          ",
    		ctx
    	});

    	return block;
    }

    // (50:10) <ListItem bind:value {...item} id={id(item)} {dense} on:change>
    function create_default_slot$2(ctx) {
    	let t_value = /*item*/ ctx[6].text + "";
    	let t;

    	const block = {
    		c: function create() {
    			t = text(t_value);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, t, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty & /*items*/ 2 && t_value !== (t_value = /*item*/ ctx[6].text + "")) set_data_dev(t, t_value);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot$2.name,
    		type: "slot",
    		source: "(50:10) <ListItem bind:value {...item} id={id(item)} {dense} on:change>",
    		ctx
    	});

    	return block;
    }

    // (48:47)          
    function fallback_block$2(ctx) {
    	let a;
    	let listitem;
    	let updating_value;
    	let a_href_value;
    	let t;
    	let current;
    	const listitem_spread_levels = [/*item*/ ctx[6], { id: id(/*item*/ ctx[6]) }, { dense: /*dense*/ ctx[2] }];

    	function listitem_value_binding(value) {
    		/*listitem_value_binding*/ ctx[13](value);
    	}

    	let listitem_props = {
    		$$slots: { default: [create_default_slot$2] },
    		$$scope: { ctx }
    	};

    	for (let i = 0; i < listitem_spread_levels.length; i += 1) {
    		listitem_props = assign(listitem_props, listitem_spread_levels[i]);
    	}

    	if (/*value*/ ctx[0] !== void 0) {
    		listitem_props.value = /*value*/ ctx[0];
    	}

    	listitem = new ListItem({ props: listitem_props, $$inline: true });
    	binding_callbacks.push(() => bind(listitem, "value", listitem_value_binding));
    	listitem.$on("change", /*change_handler*/ ctx[14]);

    	const block = {
    		c: function create() {
    			a = element("a");
    			create_component(listitem.$$.fragment);
    			t = space();
    			attr_dev(a, "tabindex", /*i*/ ctx[22] + 1);
    			attr_dev(a, "href", a_href_value = /*item*/ ctx[6].to);
    			add_location(a, file$a, 48, 8, 1154);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, a, anchor);
    			mount_component(listitem, a, null);
    			insert_dev(target, t, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const listitem_changes = (dirty & /*items, id, dense*/ 6)
    			? get_spread_update(listitem_spread_levels, [
    					dirty & /*items*/ 2 && get_spread_object(/*item*/ ctx[6]),
    					dirty & /*id, items*/ 2 && { id: id(/*item*/ ctx[6]) },
    					dirty & /*dense*/ 4 && { dense: /*dense*/ ctx[2] }
    				])
    			: {};

    			if (dirty & /*$$scope, items*/ 262146) {
    				listitem_changes.$$scope = { dirty, ctx };
    			}

    			if (!updating_value && dirty & /*value*/ 1) {
    				updating_value = true;
    				listitem_changes.value = /*value*/ ctx[0];
    				add_flush_callback(() => updating_value = false);
    			}

    			listitem.$set(listitem_changes);

    			if (!current || dirty & /*items*/ 2 && a_href_value !== (a_href_value = /*item*/ ctx[6].to)) {
    				attr_dev(a, "href", a_href_value);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(listitem.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(listitem.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(a);
    			destroy_component(listitem);
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: fallback_block$2.name,
    		type: "fallback",
    		source: "(48:47)          ",
    		ctx
    	});

    	return block;
    }

    // (46:2) {#each items as item, i}
    function create_each_block(ctx) {
    	let current_block_type_index;
    	let if_block;
    	let if_block_anchor;
    	let current;
    	const if_block_creators = [create_if_block$3, create_else_block$1];
    	const if_blocks = [];

    	function select_block_type(ctx, dirty) {
    		if (/*item*/ ctx[6].to !== undefined) return 0;
    		return 1;
    	}

    	current_block_type_index = select_block_type(ctx);
    	if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);

    	const block = {
    		c: function create() {
    			if_block.c();
    			if_block_anchor = empty();
    		},
    		m: function mount(target, anchor) {
    			if_blocks[current_block_type_index].m(target, anchor);
    			insert_dev(target, if_block_anchor, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			let previous_block_index = current_block_type_index;
    			current_block_type_index = select_block_type(ctx);

    			if (current_block_type_index === previous_block_index) {
    				if_blocks[current_block_type_index].p(ctx, dirty);
    			} else {
    				group_outros();

    				transition_out(if_blocks[previous_block_index], 1, 1, () => {
    					if_blocks[previous_block_index] = null;
    				});

    				check_outros();
    				if_block = if_blocks[current_block_type_index];

    				if (!if_block) {
    					if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
    					if_block.c();
    				} else {
    					if_block.p(ctx, dirty);
    				}

    				transition_in(if_block, 1);
    				if_block.m(if_block_anchor.parentNode, if_block_anchor);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(if_block);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(if_block);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if_blocks[current_block_type_index].d(detaching);
    			if (detaching) detach_dev(if_block_anchor);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_each_block.name,
    		type: "each",
    		source: "(46:2) {#each items as item, i}",
    		ctx
    	});

    	return block;
    }

    function create_fragment$b(ctx) {
    	let ul;
    	let current;
    	let each_value = /*items*/ ctx[1];
    	validate_each_argument(each_value);
    	let each_blocks = [];

    	for (let i = 0; i < each_value.length; i += 1) {
    		each_blocks[i] = create_each_block(get_each_context(ctx, each_value, i));
    	}

    	const out = i => transition_out(each_blocks[i], 1, 1, () => {
    		each_blocks[i] = null;
    	});

    	const block = {
    		c: function create() {
    			ul = element("ul");

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].c();
    			}

    			attr_dev(ul, "class", /*c*/ ctx[7]);
    			toggle_class(ul, "rounded-t-none", /*select*/ ctx[3]);
    			add_location(ul, file$a, 44, 0, 994);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, ul, anchor);

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].m(ul, null);
    			}

    			current = true;
    		},
    		p: function update(ctx, [dirty]) {
    			if (dirty & /*items, id, dense, value, $$scope, undefined, selectedClasses, itemClasses, getText*/ 262199) {
    				each_value = /*items*/ ctx[1];
    				validate_each_argument(each_value);
    				let i;

    				for (i = 0; i < each_value.length; i += 1) {
    					const child_ctx = get_each_context(ctx, each_value, i);

    					if (each_blocks[i]) {
    						each_blocks[i].p(child_ctx, dirty);
    						transition_in(each_blocks[i], 1);
    					} else {
    						each_blocks[i] = create_each_block(child_ctx);
    						each_blocks[i].c();
    						transition_in(each_blocks[i], 1);
    						each_blocks[i].m(ul, null);
    					}
    				}

    				group_outros();

    				for (i = each_value.length; i < each_blocks.length; i += 1) {
    					out(i);
    				}

    				check_outros();
    			}

    			if (!current || dirty & /*c*/ 128) {
    				attr_dev(ul, "class", /*c*/ ctx[7]);
    			}

    			if (dirty & /*c, select*/ 136) {
    				toggle_class(ul, "rounded-t-none", /*select*/ ctx[3]);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;

    			for (let i = 0; i < each_value.length; i += 1) {
    				transition_in(each_blocks[i]);
    			}

    			current = true;
    		},
    		o: function outro(local) {
    			each_blocks = each_blocks.filter(Boolean);

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				transition_out(each_blocks[i]);
    			}

    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(ul);
    			destroy_each(each_blocks, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$b.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    const classesDefault$2 = "py-2 rounded";

    function id(i) {
    	if (i.id !== undefined) return i.id;
    	if (i.value !== undefined) return i.value;
    	if (i.to !== undefined) return i.to;
    	if (i.text !== undefined) return i.text;
    	return i;
    }

    function getText(i) {
    	if (i.text !== undefined) return i.text;
    	if (i.value !== undefined) return i.value;
    	return i;
    }

    function instance$b($$self, $$props, $$invalidate) {
    	let c;
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("List", slots, ['item']);
    	let { items = [] } = $$props;
    	let { value = "" } = $$props;
    	let { dense = false } = $$props;
    	let { select = false } = $$props;
    	const level = null;
    	const text = "";
    	const item = {};
    	const to = null;
    	const selectedClasses = i => i;
    	const itemClasses = i => i;
    	let { classes = classesDefault$2 } = $$props;
    	const cb = new ClassBuilder($$props.class);

    	function listitem_value_binding(value$1) {
    		value = value$1;
    		$$invalidate(0, value);
    	}

    	function change_handler(event) {
    		bubble($$self, event);
    	}

    	function listitem_value_binding_1(value$1) {
    		value = value$1;
    		$$invalidate(0, value);
    	}

    	function change_handler_1(event) {
    		bubble($$self, event);
    	}

    	function click_handler(event) {
    		bubble($$self, event);
    	}

    	$$self.$$set = $$new_props => {
    		$$invalidate(20, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("items" in $$new_props) $$invalidate(1, items = $$new_props.items);
    		if ("value" in $$new_props) $$invalidate(0, value = $$new_props.value);
    		if ("dense" in $$new_props) $$invalidate(2, dense = $$new_props.dense);
    		if ("select" in $$new_props) $$invalidate(3, select = $$new_props.select);
    		if ("classes" in $$new_props) $$invalidate(11, classes = $$new_props.classes);
    		if ("$$scope" in $$new_props) $$invalidate(18, $$scope = $$new_props.$$scope);
    	};

    	$$self.$capture_state = () => ({
    		ClassBuilder,
    		ListItem,
    		items,
    		value,
    		dense,
    		select,
    		level,
    		text,
    		item,
    		to,
    		selectedClasses,
    		itemClasses,
    		classesDefault: classesDefault$2,
    		classes,
    		id,
    		getText,
    		cb,
    		c
    	});

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(20, $$props = assign(assign({}, $$props), $$new_props));
    		if ("items" in $$props) $$invalidate(1, items = $$new_props.items);
    		if ("value" in $$props) $$invalidate(0, value = $$new_props.value);
    		if ("dense" in $$props) $$invalidate(2, dense = $$new_props.dense);
    		if ("select" in $$props) $$invalidate(3, select = $$new_props.select);
    		if ("classes" in $$props) $$invalidate(11, classes = $$new_props.classes);
    		if ("c" in $$props) $$invalidate(7, c = $$new_props.c);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		$$invalidate(7, c = cb.flush().add(classes, true, classesDefault$2).add($$props.class).get());
    	};

    	$$props = exclude_internal_props($$props);

    	return [
    		value,
    		items,
    		dense,
    		select,
    		selectedClasses,
    		itemClasses,
    		item,
    		c,
    		level,
    		text,
    		to,
    		classes,
    		slots,
    		listitem_value_binding,
    		change_handler,
    		listitem_value_binding_1,
    		change_handler_1,
    		click_handler,
    		$$scope
    	];
    }

    class List extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(this, options, instance$b, create_fragment$b, safe_not_equal, {
    			items: 1,
    			value: 0,
    			dense: 2,
    			select: 3,
    			level: 8,
    			text: 9,
    			item: 6,
    			to: 10,
    			selectedClasses: 4,
    			itemClasses: 5,
    			classes: 11
    		});

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "List",
    			options,
    			id: create_fragment$b.name
    		});
    	}

    	get items() {
    		throw new Error("<List>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set items(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get value() {
    		throw new Error("<List>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set value(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get dense() {
    		throw new Error("<List>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set dense(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get select() {
    		throw new Error("<List>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set select(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get level() {
    		return this.$$.ctx[8];
    	}

    	set level(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get text() {
    		return this.$$.ctx[9];
    	}

    	set text(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get item() {
    		return this.$$.ctx[6];
    	}

    	set item(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get to() {
    		return this.$$.ctx[10];
    	}

    	set to(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get selectedClasses() {
    		return this.$$.ctx[4];
    	}

    	set selectedClasses(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get itemClasses() {
    		return this.$$.ctx[5];
    	}

    	set itemClasses(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get classes() {
    		throw new Error("<List>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set classes(value) {
    		throw new Error("<List>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* node_modules\smelte\src\components\TextField\Label.svelte generated by Svelte v3.38.2 */
    const file$9 = "node_modules\\smelte\\src\\components\\TextField\\Label.svelte";

    function create_fragment$a(ctx) {
    	let label;
    	let label_class_value;
    	let current;
    	const default_slot_template = /*#slots*/ ctx[16].default;
    	const default_slot = create_slot(default_slot_template, ctx, /*$$scope*/ ctx[15], null);

    	let label_levels = [
    		{
    			class: label_class_value = "" + (/*lClasses*/ ctx[0] + " " + /*$$props*/ ctx[2].class)
    		},
    		/*props*/ ctx[1]
    	];

    	let label_data = {};

    	for (let i = 0; i < label_levels.length; i += 1) {
    		label_data = assign(label_data, label_levels[i]);
    	}

    	const block = {
    		c: function create() {
    			label = element("label");
    			if (default_slot) default_slot.c();
    			set_attributes(label, label_data);
    			toggle_class(label, "svelte-r33x2y", true);
    			add_location(label, file$9, 72, 0, 1606);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, label, anchor);

    			if (default_slot) {
    				default_slot.m(label, null);
    			}

    			current = true;
    		},
    		p: function update(ctx, [dirty]) {
    			if (default_slot) {
    				if (default_slot.p && (!current || dirty & /*$$scope*/ 32768)) {
    					update_slot(default_slot, default_slot_template, ctx, /*$$scope*/ ctx[15], dirty, null, null);
    				}
    			}

    			set_attributes(label, label_data = get_spread_update(label_levels, [
    				(!current || dirty & /*lClasses, $$props*/ 5 && label_class_value !== (label_class_value = "" + (/*lClasses*/ ctx[0] + " " + /*$$props*/ ctx[2].class))) && { class: label_class_value },
    				/*props*/ ctx[1]
    			]));

    			toggle_class(label, "svelte-r33x2y", true);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(default_slot, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(default_slot, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(label);
    			if (default_slot) default_slot.d(detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$a.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$a($$self, $$props, $$invalidate) {
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("Label", slots, ['default']);
    	let { focused = false } = $$props;
    	let { error = false } = $$props;
    	let { outlined = false } = $$props;
    	let { labelOnTop = false } = $$props;
    	let { prepend = false } = $$props;
    	let { color = "primary" } = $$props;
    	let { bgColor = "white" } = $$props;
    	let { dense = false } = $$props;
    	let labelDefault = `pt-4 absolute top-0 label-transition block pb-2 px-4 pointer-events-none cursor-text`;
    	let { add = "" } = $$props;
    	let { remove = "" } = $$props;
    	let { replace = "" } = $$props;
    	let { labelClasses = labelDefault } = $$props;
    	const { border, txt } = utils(color);
    	const l = new ClassBuilder(labelClasses, labelDefault);
    	let lClasses = i => i;
    	const props = filterProps(["focused", "error", "outlined", "labelOnTop", "prepend", "color", "dense"], $$props);

    	$$self.$$set = $$new_props => {
    		$$invalidate(2, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("focused" in $$new_props) $$invalidate(3, focused = $$new_props.focused);
    		if ("error" in $$new_props) $$invalidate(4, error = $$new_props.error);
    		if ("outlined" in $$new_props) $$invalidate(5, outlined = $$new_props.outlined);
    		if ("labelOnTop" in $$new_props) $$invalidate(6, labelOnTop = $$new_props.labelOnTop);
    		if ("prepend" in $$new_props) $$invalidate(7, prepend = $$new_props.prepend);
    		if ("color" in $$new_props) $$invalidate(8, color = $$new_props.color);
    		if ("bgColor" in $$new_props) $$invalidate(9, bgColor = $$new_props.bgColor);
    		if ("dense" in $$new_props) $$invalidate(10, dense = $$new_props.dense);
    		if ("add" in $$new_props) $$invalidate(11, add = $$new_props.add);
    		if ("remove" in $$new_props) $$invalidate(12, remove = $$new_props.remove);
    		if ("replace" in $$new_props) $$invalidate(13, replace = $$new_props.replace);
    		if ("labelClasses" in $$new_props) $$invalidate(14, labelClasses = $$new_props.labelClasses);
    		if ("$$scope" in $$new_props) $$invalidate(15, $$scope = $$new_props.$$scope);
    	};

    	$$self.$capture_state = () => ({
    		utils,
    		ClassBuilder,
    		filterProps,
    		focused,
    		error,
    		outlined,
    		labelOnTop,
    		prepend,
    		color,
    		bgColor,
    		dense,
    		labelDefault,
    		add,
    		remove,
    		replace,
    		labelClasses,
    		border,
    		txt,
    		l,
    		lClasses,
    		props
    	});

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(2, $$props = assign(assign({}, $$props), $$new_props));
    		if ("focused" in $$props) $$invalidate(3, focused = $$new_props.focused);
    		if ("error" in $$props) $$invalidate(4, error = $$new_props.error);
    		if ("outlined" in $$props) $$invalidate(5, outlined = $$new_props.outlined);
    		if ("labelOnTop" in $$props) $$invalidate(6, labelOnTop = $$new_props.labelOnTop);
    		if ("prepend" in $$props) $$invalidate(7, prepend = $$new_props.prepend);
    		if ("color" in $$props) $$invalidate(8, color = $$new_props.color);
    		if ("bgColor" in $$props) $$invalidate(9, bgColor = $$new_props.bgColor);
    		if ("dense" in $$props) $$invalidate(10, dense = $$new_props.dense);
    		if ("labelDefault" in $$props) labelDefault = $$new_props.labelDefault;
    		if ("add" in $$props) $$invalidate(11, add = $$new_props.add);
    		if ("remove" in $$props) $$invalidate(12, remove = $$new_props.remove);
    		if ("replace" in $$props) $$invalidate(13, replace = $$new_props.replace);
    		if ("labelClasses" in $$props) $$invalidate(14, labelClasses = $$new_props.labelClasses);
    		if ("lClasses" in $$props) $$invalidate(0, lClasses = $$new_props.lClasses);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		if ($$self.$$.dirty & /*focused, error, labelOnTop, outlined, bgColor, prepend, dense, add, remove, replace*/ 16120) {
    			$$invalidate(0, lClasses = l.flush().add(txt(), focused && !error).add("text-error-500", focused && error).add("label-top text-xs", labelOnTop).add("text-xs", focused).remove("pt-4 pb-2 px-4 px-1 pt-0", labelOnTop && outlined).add(`ml-3 p-1 pt-0 mt-0 bg-${bgColor} dark:bg-dark-500`, labelOnTop && outlined).remove("px-4", prepend).add("pr-4 pl-10", prepend).remove("pt-4", dense).add("pt-3", dense).add(add).remove(remove).replace(replace).get());
    		}
    	};

    	$$props = exclude_internal_props($$props);

    	return [
    		lClasses,
    		props,
    		$$props,
    		focused,
    		error,
    		outlined,
    		labelOnTop,
    		prepend,
    		color,
    		bgColor,
    		dense,
    		add,
    		remove,
    		replace,
    		labelClasses,
    		$$scope,
    		slots
    	];
    }

    class Label extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(this, options, instance$a, create_fragment$a, safe_not_equal, {
    			focused: 3,
    			error: 4,
    			outlined: 5,
    			labelOnTop: 6,
    			prepend: 7,
    			color: 8,
    			bgColor: 9,
    			dense: 10,
    			add: 11,
    			remove: 12,
    			replace: 13,
    			labelClasses: 14
    		});

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Label",
    			options,
    			id: create_fragment$a.name
    		});
    	}

    	get focused() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set focused(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get error() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set error(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get outlined() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set outlined(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get labelOnTop() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set labelOnTop(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get prepend() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set prepend(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get color() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set color(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get bgColor() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set bgColor(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get dense() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set dense(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get add() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set add(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get remove() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set remove(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get replace() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set replace(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get labelClasses() {
    		throw new Error("<Label>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set labelClasses(value) {
    		throw new Error("<Label>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* node_modules\smelte\src\components\TextField\Hint.svelte generated by Svelte v3.38.2 */
    const file$8 = "node_modules\\smelte\\src\\components\\TextField\\Hint.svelte";

    function create_fragment$9(ctx) {
    	let div;
    	let html_tag;
    	let raw_value = (/*hint*/ ctx[1] || "") + "";
    	let t0;
    	let t1_value = (/*error*/ ctx[0] || "") + "";
    	let t1;
    	let div_transition;
    	let current;

    	const block = {
    		c: function create() {
    			div = element("div");
    			t0 = space();
    			t1 = text(t1_value);
    			html_tag = new HtmlTag(t0);
    			attr_dev(div, "class", /*classes*/ ctx[3]);
    			add_location(div, file$8, 35, 0, 787);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div, anchor);
    			html_tag.m(raw_value, div);
    			append_dev(div, t0);
    			append_dev(div, t1);
    			current = true;
    		},
    		p: function update(new_ctx, [dirty]) {
    			ctx = new_ctx;
    			if ((!current || dirty & /*hint*/ 2) && raw_value !== (raw_value = (/*hint*/ ctx[1] || "") + "")) html_tag.p(raw_value);
    			if ((!current || dirty & /*error*/ 1) && t1_value !== (t1_value = (/*error*/ ctx[0] || "") + "")) set_data_dev(t1, t1_value);

    			if (!current || dirty & /*classes*/ 8) {
    				attr_dev(div, "class", /*classes*/ ctx[3]);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;

    			add_render_callback(() => {
    				if (!div_transition) div_transition = create_bidirectional_transition(div, fly, /*transitionProps*/ ctx[2], true);
    				div_transition.run(1);
    			});

    			current = true;
    		},
    		o: function outro(local) {
    			if (!div_transition) div_transition = create_bidirectional_transition(div, fly, /*transitionProps*/ ctx[2], false);
    			div_transition.run(0);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div);
    			if (detaching && div_transition) div_transition.end();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$9.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$9($$self, $$props, $$invalidate) {
    	let classes;
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("Hint", slots, []);
    	let classesDefault = "text-xs py-1 pl-4 absolute left-0";
    	let { error = false } = $$props;
    	let { hint = "" } = $$props;
    	let { add = "" } = $$props;
    	let { remove = "" } = $$props;
    	let { replace = "" } = $$props;
    	let { transitionProps = { y: -10, duration: 100, easing: quadOut } } = $$props;
    	const l = new ClassBuilder($$props.class, classesDefault);
    	let Classes = i => i;
    	const props = filterProps(["error", "hint"], $$props);

    	$$self.$$set = $$new_props => {
    		$$invalidate(11, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("error" in $$new_props) $$invalidate(0, error = $$new_props.error);
    		if ("hint" in $$new_props) $$invalidate(1, hint = $$new_props.hint);
    		if ("add" in $$new_props) $$invalidate(4, add = $$new_props.add);
    		if ("remove" in $$new_props) $$invalidate(5, remove = $$new_props.remove);
    		if ("replace" in $$new_props) $$invalidate(6, replace = $$new_props.replace);
    		if ("transitionProps" in $$new_props) $$invalidate(2, transitionProps = $$new_props.transitionProps);
    	};

    	$$self.$capture_state = () => ({
    		utils,
    		ClassBuilder,
    		filterProps,
    		fly,
    		quadOut,
    		classesDefault,
    		error,
    		hint,
    		add,
    		remove,
    		replace,
    		transitionProps,
    		l,
    		Classes,
    		props,
    		classes
    	});

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(11, $$props = assign(assign({}, $$props), $$new_props));
    		if ("classesDefault" in $$props) classesDefault = $$new_props.classesDefault;
    		if ("error" in $$props) $$invalidate(0, error = $$new_props.error);
    		if ("hint" in $$props) $$invalidate(1, hint = $$new_props.hint);
    		if ("add" in $$props) $$invalidate(4, add = $$new_props.add);
    		if ("remove" in $$props) $$invalidate(5, remove = $$new_props.remove);
    		if ("replace" in $$props) $$invalidate(6, replace = $$new_props.replace);
    		if ("transitionProps" in $$props) $$invalidate(2, transitionProps = $$new_props.transitionProps);
    		if ("Classes" in $$props) Classes = $$new_props.Classes;
    		if ("classes" in $$props) $$invalidate(3, classes = $$new_props.classes);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		if ($$self.$$.dirty & /*error, hint, add, remove, replace*/ 115) {
    			$$invalidate(3, classes = l.flush().add("text-error-500", error).add("text-gray-600", hint).add(add).remove(remove).replace(replace).get());
    		}
    	};

    	$$props = exclude_internal_props($$props);
    	return [error, hint, transitionProps, classes, add, remove, replace];
    }

    class Hint extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(this, options, instance$9, create_fragment$9, safe_not_equal, {
    			error: 0,
    			hint: 1,
    			add: 4,
    			remove: 5,
    			replace: 6,
    			transitionProps: 2
    		});

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Hint",
    			options,
    			id: create_fragment$9.name
    		});
    	}

    	get error() {
    		throw new Error("<Hint>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set error(value) {
    		throw new Error("<Hint>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get hint() {
    		throw new Error("<Hint>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set hint(value) {
    		throw new Error("<Hint>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get add() {
    		throw new Error("<Hint>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set add(value) {
    		throw new Error("<Hint>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get remove() {
    		throw new Error("<Hint>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set remove(value) {
    		throw new Error("<Hint>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get replace() {
    		throw new Error("<Hint>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set replace(value) {
    		throw new Error("<Hint>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get transitionProps() {
    		throw new Error("<Hint>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set transitionProps(value) {
    		throw new Error("<Hint>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* node_modules\smelte\src\components\TextField\Underline.svelte generated by Svelte v3.38.2 */
    const file$7 = "node_modules\\smelte\\src\\components\\TextField\\Underline.svelte";

    function create_fragment$8(ctx) {
    	let div1;
    	let div0;
    	let div0_class_value;
    	let div1_class_value;

    	const block = {
    		c: function create() {
    			div1 = element("div");
    			div0 = element("div");
    			attr_dev(div0, "class", div0_class_value = "" + (null_to_empty(/*classes*/ ctx[2]) + " svelte-xd9zs6"));
    			set_style(div0, "height", "2px");
    			set_style(div0, "transition", "width .2s ease");
    			add_location(div0, file$7, 61, 2, 1133);
    			attr_dev(div1, "class", div1_class_value = "line absolute bottom-0 left-0 w-full bg-gray-600 " + /*$$props*/ ctx[3].class + " svelte-xd9zs6");
    			toggle_class(div1, "hidden", /*noUnderline*/ ctx[0] || /*outlined*/ ctx[1]);
    			add_location(div1, file$7, 58, 0, 1009);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div1, anchor);
    			append_dev(div1, div0);
    		},
    		p: function update(ctx, [dirty]) {
    			if (dirty & /*classes*/ 4 && div0_class_value !== (div0_class_value = "" + (null_to_empty(/*classes*/ ctx[2]) + " svelte-xd9zs6"))) {
    				attr_dev(div0, "class", div0_class_value);
    			}

    			if (dirty & /*$$props*/ 8 && div1_class_value !== (div1_class_value = "line absolute bottom-0 left-0 w-full bg-gray-600 " + /*$$props*/ ctx[3].class + " svelte-xd9zs6")) {
    				attr_dev(div1, "class", div1_class_value);
    			}

    			if (dirty & /*$$props, noUnderline, outlined*/ 11) {
    				toggle_class(div1, "hidden", /*noUnderline*/ ctx[0] || /*outlined*/ ctx[1]);
    			}
    		},
    		i: noop,
    		o: noop,
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div1);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$8.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$8($$self, $$props, $$invalidate) {
    	let classes;
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("Underline", slots, []);
    	let { noUnderline = false } = $$props;
    	let { outlined = false } = $$props;
    	let { focused = false } = $$props;
    	let { error = false } = $$props;
    	let { color = "primary" } = $$props;
    	let defaultClasses = `mx-auto w-0`;
    	let { add = "" } = $$props;
    	let { remove = "" } = $$props;
    	let { replace = "" } = $$props;
    	let { lineClasses = defaultClasses } = $$props;
    	const { bg, border, txt, caret } = utils(color);
    	const l = new ClassBuilder(lineClasses, defaultClasses);
    	let Classes = i => i;
    	const props = filterProps(["focused", "error", "outlined", "labelOnTop", "prepend", "bgcolor", "color"], $$props);

    	$$self.$$set = $$new_props => {
    		$$invalidate(3, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("noUnderline" in $$new_props) $$invalidate(0, noUnderline = $$new_props.noUnderline);
    		if ("outlined" in $$new_props) $$invalidate(1, outlined = $$new_props.outlined);
    		if ("focused" in $$new_props) $$invalidate(4, focused = $$new_props.focused);
    		if ("error" in $$new_props) $$invalidate(5, error = $$new_props.error);
    		if ("color" in $$new_props) $$invalidate(6, color = $$new_props.color);
    		if ("add" in $$new_props) $$invalidate(7, add = $$new_props.add);
    		if ("remove" in $$new_props) $$invalidate(8, remove = $$new_props.remove);
    		if ("replace" in $$new_props) $$invalidate(9, replace = $$new_props.replace);
    		if ("lineClasses" in $$new_props) $$invalidate(10, lineClasses = $$new_props.lineClasses);
    	};

    	$$self.$capture_state = () => ({
    		utils,
    		ClassBuilder,
    		filterProps,
    		noUnderline,
    		outlined,
    		focused,
    		error,
    		color,
    		defaultClasses,
    		add,
    		remove,
    		replace,
    		lineClasses,
    		bg,
    		border,
    		txt,
    		caret,
    		l,
    		Classes,
    		props,
    		classes
    	});

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(3, $$props = assign(assign({}, $$props), $$new_props));
    		if ("noUnderline" in $$props) $$invalidate(0, noUnderline = $$new_props.noUnderline);
    		if ("outlined" in $$props) $$invalidate(1, outlined = $$new_props.outlined);
    		if ("focused" in $$props) $$invalidate(4, focused = $$new_props.focused);
    		if ("error" in $$props) $$invalidate(5, error = $$new_props.error);
    		if ("color" in $$props) $$invalidate(6, color = $$new_props.color);
    		if ("defaultClasses" in $$props) defaultClasses = $$new_props.defaultClasses;
    		if ("add" in $$props) $$invalidate(7, add = $$new_props.add);
    		if ("remove" in $$props) $$invalidate(8, remove = $$new_props.remove);
    		if ("replace" in $$props) $$invalidate(9, replace = $$new_props.replace);
    		if ("lineClasses" in $$props) $$invalidate(10, lineClasses = $$new_props.lineClasses);
    		if ("Classes" in $$props) Classes = $$new_props.Classes;
    		if ("classes" in $$props) $$invalidate(2, classes = $$new_props.classes);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		if ($$self.$$.dirty & /*focused, error, add, remove, replace*/ 944) {
    			$$invalidate(2, classes = l.flush().add(txt(), focused && !error).add("bg-error-500", error).add("w-full", focused || error).add(bg(), focused).add(add).remove(remove).replace(replace).get());
    		}
    	};

    	$$props = exclude_internal_props($$props);

    	return [
    		noUnderline,
    		outlined,
    		classes,
    		$$props,
    		focused,
    		error,
    		color,
    		add,
    		remove,
    		replace,
    		lineClasses
    	];
    }

    class Underline extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(this, options, instance$8, create_fragment$8, safe_not_equal, {
    			noUnderline: 0,
    			outlined: 1,
    			focused: 4,
    			error: 5,
    			color: 6,
    			add: 7,
    			remove: 8,
    			replace: 9,
    			lineClasses: 10
    		});

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Underline",
    			options,
    			id: create_fragment$8.name
    		});
    	}

    	get noUnderline() {
    		throw new Error("<Underline>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set noUnderline(value) {
    		throw new Error("<Underline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get outlined() {
    		throw new Error("<Underline>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set outlined(value) {
    		throw new Error("<Underline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get focused() {
    		throw new Error("<Underline>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set focused(value) {
    		throw new Error("<Underline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get error() {
    		throw new Error("<Underline>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set error(value) {
    		throw new Error("<Underline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get color() {
    		throw new Error("<Underline>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set color(value) {
    		throw new Error("<Underline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get add() {
    		throw new Error("<Underline>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set add(value) {
    		throw new Error("<Underline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get remove() {
    		throw new Error("<Underline>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set remove(value) {
    		throw new Error("<Underline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get replace() {
    		throw new Error("<Underline>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set replace(value) {
    		throw new Error("<Underline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get lineClasses() {
    		throw new Error("<Underline>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set lineClasses(value) {
    		throw new Error("<Underline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* node_modules\smelte\src\components\TextField\TextField.svelte generated by Svelte v3.38.2 */
    const file$6 = "node_modules\\smelte\\src\\components\\TextField\\TextField.svelte";
    const get_prepend_slot_changes = dirty => ({});
    const get_prepend_slot_context = ctx => ({});
    const get_append_slot_changes = dirty => ({});
    const get_append_slot_context = ctx => ({});
    const get_label_slot_changes = dirty => ({});
    const get_label_slot_context = ctx => ({});

    // (139:2) {#if label}
    function create_if_block_6(ctx) {
    	let current;
    	const label_slot_template = /*#slots*/ ctx[40].label;
    	const label_slot = create_slot(label_slot_template, ctx, /*$$scope*/ ctx[69], get_label_slot_context);
    	const label_slot_or_fallback = label_slot || fallback_block_2(ctx);

    	const block = {
    		c: function create() {
    			if (label_slot_or_fallback) label_slot_or_fallback.c();
    		},
    		m: function mount(target, anchor) {
    			if (label_slot_or_fallback) {
    				label_slot_or_fallback.m(target, anchor);
    			}

    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			if (label_slot) {
    				if (label_slot.p && (!current || dirty[2] & /*$$scope*/ 128)) {
    					update_slot(label_slot, label_slot_template, ctx, /*$$scope*/ ctx[69], dirty, get_label_slot_changes, get_label_slot_context);
    				}
    			} else {
    				if (label_slot_or_fallback && label_slot_or_fallback.p && dirty[0] & /*labelOnTop, focused, error, outlined, prepend, color, bgColor, dense, label*/ 33952078) {
    					label_slot_or_fallback.p(ctx, dirty);
    				}
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(label_slot_or_fallback, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(label_slot_or_fallback, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (label_slot_or_fallback) label_slot_or_fallback.d(detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_6.name,
    		type: "if",
    		source: "(139:2) {#if label}",
    		ctx
    	});

    	return block;
    }

    // (141:4) <Label       {labelOnTop}       {focused}       {error}       {outlined}       {prepend}       {color}       {bgColor}       dense={dense && !outlined}     >
    function create_default_slot_2(ctx) {
    	let t;

    	const block = {
    		c: function create() {
    			t = text(/*label*/ ctx[3]);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, t, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*label*/ 8) set_data_dev(t, /*label*/ ctx[3]);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot_2.name,
    		type: "slot",
    		source: "(141:4) <Label       {labelOnTop}       {focused}       {error}       {outlined}       {prepend}       {color}       {bgColor}       dense={dense && !outlined}     >",
    		ctx
    	});

    	return block;
    }

    // (140:21)      
    function fallback_block_2(ctx) {
    	let label_1;
    	let current;

    	label_1 = new Label({
    			props: {
    				labelOnTop: /*labelOnTop*/ ctx[25],
    				focused: /*focused*/ ctx[1],
    				error: /*error*/ ctx[6],
    				outlined: /*outlined*/ ctx[2],
    				prepend: /*prepend*/ ctx[8],
    				color: /*color*/ ctx[17],
    				bgColor: /*bgColor*/ ctx[18],
    				dense: /*dense*/ ctx[12] && !/*outlined*/ ctx[2],
    				$$slots: { default: [create_default_slot_2] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	const block = {
    		c: function create() {
    			create_component(label_1.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(label_1, target, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const label_1_changes = {};
    			if (dirty[0] & /*labelOnTop*/ 33554432) label_1_changes.labelOnTop = /*labelOnTop*/ ctx[25];
    			if (dirty[0] & /*focused*/ 2) label_1_changes.focused = /*focused*/ ctx[1];
    			if (dirty[0] & /*error*/ 64) label_1_changes.error = /*error*/ ctx[6];
    			if (dirty[0] & /*outlined*/ 4) label_1_changes.outlined = /*outlined*/ ctx[2];
    			if (dirty[0] & /*prepend*/ 256) label_1_changes.prepend = /*prepend*/ ctx[8];
    			if (dirty[0] & /*color*/ 131072) label_1_changes.color = /*color*/ ctx[17];
    			if (dirty[0] & /*bgColor*/ 262144) label_1_changes.bgColor = /*bgColor*/ ctx[18];
    			if (dirty[0] & /*dense, outlined*/ 4100) label_1_changes.dense = /*dense*/ ctx[12] && !/*outlined*/ ctx[2];

    			if (dirty[0] & /*label*/ 8 | dirty[2] & /*$$scope*/ 128) {
    				label_1_changes.$$scope = { dirty, ctx };
    			}

    			label_1.$set(label_1_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(label_1.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(label_1.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(label_1, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: fallback_block_2.name,
    		type: "fallback",
    		source: "(140:21)      ",
    		ctx
    	});

    	return block;
    }

    // (191:36) 
    function create_if_block_5(ctx) {
    	let input;
    	let mounted;
    	let dispose;

    	const block = {
    		c: function create() {
    			input = element("input");
    			input.readOnly = true;
    			attr_dev(input, "class", /*iClasses*/ ctx[26]);
    			input.disabled = /*disabled*/ ctx[20];
    			input.value = /*value*/ ctx[0];
    			add_location(input, file$6, 191, 4, 4933);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, input, anchor);

    			if (!mounted) {
    				dispose = [
    					listen_dev(input, "change", /*change_handler_2*/ ctx[57], false, false, false),
    					listen_dev(input, "input", /*input_handler_2*/ ctx[58], false, false, false),
    					listen_dev(input, "keydown", /*keydown_handler_2*/ ctx[59], false, false, false),
    					listen_dev(input, "keypress", /*keypress_handler_2*/ ctx[60], false, false, false),
    					listen_dev(input, "keyup", /*keyup_handler_2*/ ctx[61], false, false, false),
    					listen_dev(input, "click", /*click_handler_2*/ ctx[62], false, false, false),
    					listen_dev(input, "blur", /*blur_handler_2*/ ctx[63], false, false, false),
    					listen_dev(input, "focus", /*focus_handler_2*/ ctx[64], false, false, false)
    				];

    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*iClasses*/ 67108864) {
    				attr_dev(input, "class", /*iClasses*/ ctx[26]);
    			}

    			if (dirty[0] & /*disabled*/ 1048576) {
    				prop_dev(input, "disabled", /*disabled*/ ctx[20]);
    			}

    			if (dirty[0] & /*value*/ 1 && input.value !== /*value*/ ctx[0]) {
    				prop_dev(input, "value", /*value*/ ctx[0]);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(input);
    			mounted = false;
    			run_all(dispose);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_5.name,
    		type: "if",
    		source: "(191:36) ",
    		ctx
    	});

    	return block;
    }

    // (172:32) 
    function create_if_block_4(ctx) {
    	let textarea_1;
    	let textarea_1_placeholder_value;
    	let mounted;
    	let dispose;

    	let textarea_1_levels = [
    		{ rows: /*rows*/ ctx[10] },
    		{ "aria-label": /*label*/ ctx[3] },
    		{ class: /*iClasses*/ ctx[26] },
    		{ disabled: /*disabled*/ ctx[20] },
    		/*props*/ ctx[29],
    		{
    			placeholder: textarea_1_placeholder_value = !/*value*/ ctx[0] ? /*placeholder*/ ctx[4] : ""
    		}
    	];

    	let textarea_1_data = {};

    	for (let i = 0; i < textarea_1_levels.length; i += 1) {
    		textarea_1_data = assign(textarea_1_data, textarea_1_levels[i]);
    	}

    	const block = {
    		c: function create() {
    			textarea_1 = element("textarea");
    			set_attributes(textarea_1, textarea_1_data);
    			add_location(textarea_1, file$6, 172, 4, 4535);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, textarea_1, anchor);
    			set_input_value(textarea_1, /*value*/ ctx[0]);

    			if (!mounted) {
    				dispose = [
    					listen_dev(textarea_1, "change", /*change_handler_1*/ ctx[49], false, false, false),
    					listen_dev(textarea_1, "input", /*input_handler_1*/ ctx[50], false, false, false),
    					listen_dev(textarea_1, "keydown", /*keydown_handler_1*/ ctx[51], false, false, false),
    					listen_dev(textarea_1, "keypress", /*keypress_handler_1*/ ctx[52], false, false, false),
    					listen_dev(textarea_1, "keyup", /*keyup_handler_1*/ ctx[53], false, false, false),
    					listen_dev(textarea_1, "click", /*click_handler_1*/ ctx[54], false, false, false),
    					listen_dev(textarea_1, "focus", /*focus_handler_1*/ ctx[55], false, false, false),
    					listen_dev(textarea_1, "blur", /*blur_handler_1*/ ctx[56], false, false, false),
    					listen_dev(textarea_1, "input", /*textarea_1_input_handler*/ ctx[66]),
    					listen_dev(textarea_1, "focus", /*toggleFocused*/ ctx[28], false, false, false),
    					listen_dev(textarea_1, "blur", /*toggleFocused*/ ctx[28], false, false, false)
    				];

    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			set_attributes(textarea_1, textarea_1_data = get_spread_update(textarea_1_levels, [
    				dirty[0] & /*rows*/ 1024 && { rows: /*rows*/ ctx[10] },
    				dirty[0] & /*label*/ 8 && { "aria-label": /*label*/ ctx[3] },
    				dirty[0] & /*iClasses*/ 67108864 && { class: /*iClasses*/ ctx[26] },
    				dirty[0] & /*disabled*/ 1048576 && { disabled: /*disabled*/ ctx[20] },
    				/*props*/ ctx[29],
    				dirty[0] & /*value, placeholder*/ 17 && textarea_1_placeholder_value !== (textarea_1_placeholder_value = !/*value*/ ctx[0] ? /*placeholder*/ ctx[4] : "") && {
    					placeholder: textarea_1_placeholder_value
    				}
    			]));

    			if (dirty[0] & /*value*/ 1) {
    				set_input_value(textarea_1, /*value*/ ctx[0]);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(textarea_1);
    			mounted = false;
    			run_all(dispose);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_4.name,
    		type: "if",
    		source: "(172:32) ",
    		ctx
    	});

    	return block;
    }

    // (154:2) {#if (!textarea && !select) || autocomplete}
    function create_if_block_3(ctx) {
    	let input;
    	let input_placeholder_value;
    	let mounted;
    	let dispose;

    	let input_levels = [
    		{ "aria-label": /*label*/ ctx[3] },
    		{ class: /*iClasses*/ ctx[26] },
    		{ disabled: /*disabled*/ ctx[20] },
    		/*props*/ ctx[29],
    		{
    			placeholder: input_placeholder_value = !/*value*/ ctx[0] ? /*placeholder*/ ctx[4] : ""
    		}
    	];

    	let input_data = {};

    	for (let i = 0; i < input_levels.length; i += 1) {
    		input_data = assign(input_data, input_levels[i]);
    	}

    	const block = {
    		c: function create() {
    			input = element("input");
    			set_attributes(input, input_data);
    			add_location(input, file$6, 154, 4, 4157);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, input, anchor);
    			set_input_value(input, /*value*/ ctx[0]);

    			if (!mounted) {
    				dispose = [
    					listen_dev(input, "focus", /*toggleFocused*/ ctx[28], false, false, false),
    					listen_dev(input, "blur", /*toggleFocused*/ ctx[28], false, false, false),
    					listen_dev(input, "blur", /*blur_handler*/ ctx[41], false, false, false),
    					listen_dev(input, "input", /*input_input_handler*/ ctx[65]),
    					listen_dev(input, "change", /*change_handler*/ ctx[42], false, false, false),
    					listen_dev(input, "input", /*input_handler*/ ctx[43], false, false, false),
    					listen_dev(input, "keydown", /*keydown_handler*/ ctx[44], false, false, false),
    					listen_dev(input, "keypress", /*keypress_handler*/ ctx[45], false, false, false),
    					listen_dev(input, "keyup", /*keyup_handler*/ ctx[46], false, false, false),
    					listen_dev(input, "click", /*click_handler*/ ctx[47], false, false, false),
    					listen_dev(input, "focus", /*focus_handler*/ ctx[48], false, false, false)
    				];

    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			set_attributes(input, input_data = get_spread_update(input_levels, [
    				dirty[0] & /*label*/ 8 && { "aria-label": /*label*/ ctx[3] },
    				dirty[0] & /*iClasses*/ 67108864 && { class: /*iClasses*/ ctx[26] },
    				dirty[0] & /*disabled*/ 1048576 && { disabled: /*disabled*/ ctx[20] },
    				/*props*/ ctx[29],
    				dirty[0] & /*value, placeholder*/ 17 && input_placeholder_value !== (input_placeholder_value = !/*value*/ ctx[0] ? /*placeholder*/ ctx[4] : "") && { placeholder: input_placeholder_value }
    			]));

    			if (dirty[0] & /*value*/ 1 && input.value !== /*value*/ ctx[0]) {
    				set_input_value(input, /*value*/ ctx[0]);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(input);
    			mounted = false;
    			run_all(dispose);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_3.name,
    		type: "if",
    		source: "(154:2) {#if (!textarea && !select) || autocomplete}",
    		ctx
    	});

    	return block;
    }

    // (207:2) {#if append}
    function create_if_block_2$1(ctx) {
    	let div;
    	let current;
    	let mounted;
    	let dispose;
    	const append_slot_template = /*#slots*/ ctx[40].append;
    	const append_slot = create_slot(append_slot_template, ctx, /*$$scope*/ ctx[69], get_append_slot_context);
    	const append_slot_or_fallback = append_slot || fallback_block_1$1(ctx);

    	const block = {
    		c: function create() {
    			div = element("div");
    			if (append_slot_or_fallback) append_slot_or_fallback.c();
    			attr_dev(div, "class", /*aClasses*/ ctx[22]);
    			add_location(div, file$6, 207, 4, 5167);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div, anchor);

    			if (append_slot_or_fallback) {
    				append_slot_or_fallback.m(div, null);
    			}

    			current = true;

    			if (!mounted) {
    				dispose = listen_dev(div, "click", /*click_handler_3*/ ctx[67], false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (append_slot) {
    				if (append_slot.p && (!current || dirty[2] & /*$$scope*/ 128)) {
    					update_slot(append_slot, append_slot_template, ctx, /*$$scope*/ ctx[69], dirty, get_append_slot_changes, get_append_slot_context);
    				}
    			} else {
    				if (append_slot_or_fallback && append_slot_or_fallback.p && dirty[0] & /*appendReverse, focused, iconClass, append*/ 557186) {
    					append_slot_or_fallback.p(ctx, dirty);
    				}
    			}

    			if (!current || dirty[0] & /*aClasses*/ 4194304) {
    				attr_dev(div, "class", /*aClasses*/ ctx[22]);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(append_slot_or_fallback, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(append_slot_or_fallback, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div);
    			if (append_slot_or_fallback) append_slot_or_fallback.d(detaching);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_2$1.name,
    		type: "if",
    		source: "(207:2) {#if append}",
    		ctx
    	});

    	return block;
    }

    // (213:8) <Icon           reverse={appendReverse}           class="{focused ? txt() : ""} {iconClass}"         >
    function create_default_slot_1(ctx) {
    	let t;

    	const block = {
    		c: function create() {
    			t = text(/*append*/ ctx[7]);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, t, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*append*/ 128) set_data_dev(t, /*append*/ ctx[7]);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot_1.name,
    		type: "slot",
    		source: "(213:8) <Icon           reverse={appendReverse}           class=\\\"{focused ? txt() : \\\"\\\"} {iconClass}\\\"         >",
    		ctx
    	});

    	return block;
    }

    // (212:26)          
    function fallback_block_1$1(ctx) {
    	let icon;
    	let current;

    	icon = new Icon({
    			props: {
    				reverse: /*appendReverse*/ ctx[15],
    				class: "" + ((/*focused*/ ctx[1] ? /*txt*/ ctx[27]() : "") + " " + /*iconClass*/ ctx[19]),
    				$$slots: { default: [create_default_slot_1] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	const block = {
    		c: function create() {
    			create_component(icon.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(icon, target, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const icon_changes = {};
    			if (dirty[0] & /*appendReverse*/ 32768) icon_changes.reverse = /*appendReverse*/ ctx[15];
    			if (dirty[0] & /*focused, iconClass*/ 524290) icon_changes.class = "" + ((/*focused*/ ctx[1] ? /*txt*/ ctx[27]() : "") + " " + /*iconClass*/ ctx[19]);

    			if (dirty[0] & /*append*/ 128 | dirty[2] & /*$$scope*/ 128) {
    				icon_changes.$$scope = { dirty, ctx };
    			}

    			icon.$set(icon_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(icon.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(icon.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(icon, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: fallback_block_1$1.name,
    		type: "fallback",
    		source: "(212:26)          ",
    		ctx
    	});

    	return block;
    }

    // (223:2) {#if prepend}
    function create_if_block_1$1(ctx) {
    	let div;
    	let current;
    	let mounted;
    	let dispose;
    	const prepend_slot_template = /*#slots*/ ctx[40].prepend;
    	const prepend_slot = create_slot(prepend_slot_template, ctx, /*$$scope*/ ctx[69], get_prepend_slot_context);
    	const prepend_slot_or_fallback = prepend_slot || fallback_block$1(ctx);

    	const block = {
    		c: function create() {
    			div = element("div");
    			if (prepend_slot_or_fallback) prepend_slot_or_fallback.c();
    			attr_dev(div, "class", /*pClasses*/ ctx[23]);
    			add_location(div, file$6, 223, 4, 5476);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div, anchor);

    			if (prepend_slot_or_fallback) {
    				prepend_slot_or_fallback.m(div, null);
    			}

    			current = true;

    			if (!mounted) {
    				dispose = listen_dev(div, "click", /*click_handler_4*/ ctx[68], false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (prepend_slot) {
    				if (prepend_slot.p && (!current || dirty[2] & /*$$scope*/ 128)) {
    					update_slot(prepend_slot, prepend_slot_template, ctx, /*$$scope*/ ctx[69], dirty, get_prepend_slot_changes, get_prepend_slot_context);
    				}
    			} else {
    				if (prepend_slot_or_fallback && prepend_slot_or_fallback.p && dirty[0] & /*prependReverse, focused, iconClass, prepend*/ 590082) {
    					prepend_slot_or_fallback.p(ctx, dirty);
    				}
    			}

    			if (!current || dirty[0] & /*pClasses*/ 8388608) {
    				attr_dev(div, "class", /*pClasses*/ ctx[23]);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(prepend_slot_or_fallback, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(prepend_slot_or_fallback, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div);
    			if (prepend_slot_or_fallback) prepend_slot_or_fallback.d(detaching);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_1$1.name,
    		type: "if",
    		source: "(223:2) {#if prepend}",
    		ctx
    	});

    	return block;
    }

    // (229:8) <Icon           reverse={prependReverse}           class="{focused ? txt() : ""} {iconClass}"         >
    function create_default_slot$1(ctx) {
    	let t;

    	const block = {
    		c: function create() {
    			t = text(/*prepend*/ ctx[8]);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, t, anchor);
    		},
    		p: function update(ctx, dirty) {
    			if (dirty[0] & /*prepend*/ 256) set_data_dev(t, /*prepend*/ ctx[8]);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot$1.name,
    		type: "slot",
    		source: "(229:8) <Icon           reverse={prependReverse}           class=\\\"{focused ? txt() : \\\"\\\"} {iconClass}\\\"         >",
    		ctx
    	});

    	return block;
    }

    // (228:27)          
    function fallback_block$1(ctx) {
    	let icon;
    	let current;

    	icon = new Icon({
    			props: {
    				reverse: /*prependReverse*/ ctx[16],
    				class: "" + ((/*focused*/ ctx[1] ? /*txt*/ ctx[27]() : "") + " " + /*iconClass*/ ctx[19]),
    				$$slots: { default: [create_default_slot$1] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	const block = {
    		c: function create() {
    			create_component(icon.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(icon, target, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const icon_changes = {};
    			if (dirty[0] & /*prependReverse*/ 65536) icon_changes.reverse = /*prependReverse*/ ctx[16];
    			if (dirty[0] & /*focused, iconClass*/ 524290) icon_changes.class = "" + ((/*focused*/ ctx[1] ? /*txt*/ ctx[27]() : "") + " " + /*iconClass*/ ctx[19]);

    			if (dirty[0] & /*prepend*/ 256 | dirty[2] & /*$$scope*/ 128) {
    				icon_changes.$$scope = { dirty, ctx };
    			}

    			icon.$set(icon_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(icon.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(icon.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(icon, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: fallback_block$1.name,
    		type: "fallback",
    		source: "(228:27)          ",
    		ctx
    	});

    	return block;
    }

    // (245:2) {#if showHint}
    function create_if_block$2(ctx) {
    	let hint_1;
    	let current;

    	hint_1 = new Hint({
    			props: {
    				error: /*error*/ ctx[6],
    				hint: /*hint*/ ctx[5]
    			},
    			$$inline: true
    		});

    	const block = {
    		c: function create() {
    			create_component(hint_1.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(hint_1, target, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const hint_1_changes = {};
    			if (dirty[0] & /*error*/ 64) hint_1_changes.error = /*error*/ ctx[6];
    			if (dirty[0] & /*hint*/ 32) hint_1_changes.hint = /*hint*/ ctx[5];
    			hint_1.$set(hint_1_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(hint_1.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(hint_1.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(hint_1, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block$2.name,
    		type: "if",
    		source: "(245:2) {#if showHint}",
    		ctx
    	});

    	return block;
    }

    function create_fragment$7(ctx) {
    	let div;
    	let t0;
    	let t1;
    	let t2;
    	let t3;
    	let underline;
    	let t4;
    	let current;
    	let if_block0 = /*label*/ ctx[3] && create_if_block_6(ctx);

    	function select_block_type(ctx, dirty) {
    		if (!/*textarea*/ ctx[9] && !/*select*/ ctx[11] || /*autocomplete*/ ctx[13]) return create_if_block_3;
    		if (/*textarea*/ ctx[9] && !/*select*/ ctx[11]) return create_if_block_4;
    		if (/*select*/ ctx[11] && !/*autocomplete*/ ctx[13]) return create_if_block_5;
    	}

    	let current_block_type = select_block_type(ctx);
    	let if_block1 = current_block_type && current_block_type(ctx);
    	let if_block2 = /*append*/ ctx[7] && create_if_block_2$1(ctx);
    	let if_block3 = /*prepend*/ ctx[8] && create_if_block_1$1(ctx);

    	underline = new Underline({
    			props: {
    				noUnderline: /*noUnderline*/ ctx[14],
    				outlined: /*outlined*/ ctx[2],
    				focused: /*focused*/ ctx[1],
    				error: /*error*/ ctx[6]
    			},
    			$$inline: true
    		});

    	let if_block4 = /*showHint*/ ctx[24] && create_if_block$2(ctx);

    	const block = {
    		c: function create() {
    			div = element("div");
    			if (if_block0) if_block0.c();
    			t0 = space();
    			if (if_block1) if_block1.c();
    			t1 = space();
    			if (if_block2) if_block2.c();
    			t2 = space();
    			if (if_block3) if_block3.c();
    			t3 = space();
    			create_component(underline.$$.fragment);
    			t4 = space();
    			if (if_block4) if_block4.c();
    			attr_dev(div, "class", /*wClasses*/ ctx[21]);
    			add_location(div, file$6, 137, 0, 3851);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div, anchor);
    			if (if_block0) if_block0.m(div, null);
    			append_dev(div, t0);
    			if (if_block1) if_block1.m(div, null);
    			append_dev(div, t1);
    			if (if_block2) if_block2.m(div, null);
    			append_dev(div, t2);
    			if (if_block3) if_block3.m(div, null);
    			append_dev(div, t3);
    			mount_component(underline, div, null);
    			append_dev(div, t4);
    			if (if_block4) if_block4.m(div, null);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			if (/*label*/ ctx[3]) {
    				if (if_block0) {
    					if_block0.p(ctx, dirty);

    					if (dirty[0] & /*label*/ 8) {
    						transition_in(if_block0, 1);
    					}
    				} else {
    					if_block0 = create_if_block_6(ctx);
    					if_block0.c();
    					transition_in(if_block0, 1);
    					if_block0.m(div, t0);
    				}
    			} else if (if_block0) {
    				group_outros();

    				transition_out(if_block0, 1, 1, () => {
    					if_block0 = null;
    				});

    				check_outros();
    			}

    			if (current_block_type === (current_block_type = select_block_type(ctx)) && if_block1) {
    				if_block1.p(ctx, dirty);
    			} else {
    				if (if_block1) if_block1.d(1);
    				if_block1 = current_block_type && current_block_type(ctx);

    				if (if_block1) {
    					if_block1.c();
    					if_block1.m(div, t1);
    				}
    			}

    			if (/*append*/ ctx[7]) {
    				if (if_block2) {
    					if_block2.p(ctx, dirty);

    					if (dirty[0] & /*append*/ 128) {
    						transition_in(if_block2, 1);
    					}
    				} else {
    					if_block2 = create_if_block_2$1(ctx);
    					if_block2.c();
    					transition_in(if_block2, 1);
    					if_block2.m(div, t2);
    				}
    			} else if (if_block2) {
    				group_outros();

    				transition_out(if_block2, 1, 1, () => {
    					if_block2 = null;
    				});

    				check_outros();
    			}

    			if (/*prepend*/ ctx[8]) {
    				if (if_block3) {
    					if_block3.p(ctx, dirty);

    					if (dirty[0] & /*prepend*/ 256) {
    						transition_in(if_block3, 1);
    					}
    				} else {
    					if_block3 = create_if_block_1$1(ctx);
    					if_block3.c();
    					transition_in(if_block3, 1);
    					if_block3.m(div, t3);
    				}
    			} else if (if_block3) {
    				group_outros();

    				transition_out(if_block3, 1, 1, () => {
    					if_block3 = null;
    				});

    				check_outros();
    			}

    			const underline_changes = {};
    			if (dirty[0] & /*noUnderline*/ 16384) underline_changes.noUnderline = /*noUnderline*/ ctx[14];
    			if (dirty[0] & /*outlined*/ 4) underline_changes.outlined = /*outlined*/ ctx[2];
    			if (dirty[0] & /*focused*/ 2) underline_changes.focused = /*focused*/ ctx[1];
    			if (dirty[0] & /*error*/ 64) underline_changes.error = /*error*/ ctx[6];
    			underline.$set(underline_changes);

    			if (/*showHint*/ ctx[24]) {
    				if (if_block4) {
    					if_block4.p(ctx, dirty);

    					if (dirty[0] & /*showHint*/ 16777216) {
    						transition_in(if_block4, 1);
    					}
    				} else {
    					if_block4 = create_if_block$2(ctx);
    					if_block4.c();
    					transition_in(if_block4, 1);
    					if_block4.m(div, null);
    				}
    			} else if (if_block4) {
    				group_outros();

    				transition_out(if_block4, 1, 1, () => {
    					if_block4 = null;
    				});

    				check_outros();
    			}

    			if (!current || dirty[0] & /*wClasses*/ 2097152) {
    				attr_dev(div, "class", /*wClasses*/ ctx[21]);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(if_block0);
    			transition_in(if_block2);
    			transition_in(if_block3);
    			transition_in(underline.$$.fragment, local);
    			transition_in(if_block4);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(if_block0);
    			transition_out(if_block2);
    			transition_out(if_block3);
    			transition_out(underline.$$.fragment, local);
    			transition_out(if_block4);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div);
    			if (if_block0) if_block0.d();

    			if (if_block1) {
    				if_block1.d();
    			}

    			if (if_block2) if_block2.d();
    			if (if_block3) if_block3.d();
    			destroy_component(underline);
    			if (if_block4) if_block4.d();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$7.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    const inputDefault = "pb-2 pt-6 px-4 rounded-t text-black dark:text-gray-100 w-full";
    const classesDefault$1 = "mt-2 mb-6 relative text-gray-600 dark:text-gray-100";
    const appendDefault = "absolute right-0 top-0 pb-2 pr-4 pt-4 text-gray-700 z-10";
    const prependDefault = "absolute left-0 top-0 pb-2 pl-2 pt-4 text-xs text-gray-700 z-10";

    function instance$7($$self, $$props, $$invalidate) {
    	let showHint;
    	let labelOnTop;
    	let iClasses;
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("TextField", slots, ['label','append','prepend']);
    	let { outlined = false } = $$props;
    	let { value = null } = $$props;
    	let { label = "" } = $$props;
    	let { placeholder = "" } = $$props;
    	let { hint = "" } = $$props;
    	let { error = false } = $$props;
    	let { append = "" } = $$props;
    	let { prepend = "" } = $$props;
    	let { persistentHint = false } = $$props;
    	let { textarea = false } = $$props;
    	let { rows = 5 } = $$props;
    	let { select = false } = $$props;
    	let { dense = false } = $$props;
    	let { autocomplete = false } = $$props;
    	let { noUnderline = false } = $$props;
    	let { appendReverse = false } = $$props;
    	let { prependReverse = false } = $$props;
    	let { color = "primary" } = $$props;
    	let { bgColor = "white" } = $$props;
    	let { iconClass = "" } = $$props;
    	let { disabled = false } = $$props;
    	let { add = "" } = $$props;
    	let { remove = "" } = $$props;
    	let { replace = "" } = $$props;
    	let { inputClasses = inputDefault } = $$props;
    	let { classes = classesDefault$1 } = $$props;
    	let { appendClasses = appendDefault } = $$props;
    	let { prependClasses = prependDefault } = $$props;
    	const { bg, border, txt, caret } = utils(color);
    	const cb = new ClassBuilder(inputClasses, inputDefault);
    	const ccb = new ClassBuilder(classes, classesDefault$1);
    	const acb = new ClassBuilder(appendClasses, appendDefault);
    	const pcb = new ClassBuilder(prependClasses, prependDefault);

    	let { extend = () => {
    		
    	} } = $$props;

    	let { focused = false } = $$props;
    	let wClasses = i => i;
    	let aClasses = i => i;
    	let pClasses = i => i;

    	function toggleFocused() {
    		$$invalidate(1, focused = !focused);
    	}

    	const props = filterProps(
    		[
    			"outlined",
    			"label",
    			"placeholder",
    			"hint",
    			"error",
    			"append",
    			"prepend",
    			"persistentHint",
    			"textarea",
    			"rows",
    			"select",
    			"autocomplete",
    			"noUnderline",
    			"appendReverse",
    			"prependReverse",
    			"color",
    			"bgColor",
    			"disabled",
    			"replace",
    			"remove",
    			"small"
    		],
    		$$props
    	);

    	const dispatch = createEventDispatcher();

    	function blur_handler(event) {
    		bubble($$self, event);
    	}

    	function change_handler(event) {
    		bubble($$self, event);
    	}

    	function input_handler(event) {
    		bubble($$self, event);
    	}

    	function keydown_handler(event) {
    		bubble($$self, event);
    	}

    	function keypress_handler(event) {
    		bubble($$self, event);
    	}

    	function keyup_handler(event) {
    		bubble($$self, event);
    	}

    	function click_handler(event) {
    		bubble($$self, event);
    	}

    	function focus_handler(event) {
    		bubble($$self, event);
    	}

    	function change_handler_1(event) {
    		bubble($$self, event);
    	}

    	function input_handler_1(event) {
    		bubble($$self, event);
    	}

    	function keydown_handler_1(event) {
    		bubble($$self, event);
    	}

    	function keypress_handler_1(event) {
    		bubble($$self, event);
    	}

    	function keyup_handler_1(event) {
    		bubble($$self, event);
    	}

    	function click_handler_1(event) {
    		bubble($$self, event);
    	}

    	function focus_handler_1(event) {
    		bubble($$self, event);
    	}

    	function blur_handler_1(event) {
    		bubble($$self, event);
    	}

    	function change_handler_2(event) {
    		bubble($$self, event);
    	}

    	function input_handler_2(event) {
    		bubble($$self, event);
    	}

    	function keydown_handler_2(event) {
    		bubble($$self, event);
    	}

    	function keypress_handler_2(event) {
    		bubble($$self, event);
    	}

    	function keyup_handler_2(event) {
    		bubble($$self, event);
    	}

    	function click_handler_2(event) {
    		bubble($$self, event);
    	}

    	function blur_handler_2(event) {
    		bubble($$self, event);
    	}

    	function focus_handler_2(event) {
    		bubble($$self, event);
    	}

    	function input_input_handler() {
    		value = this.value;
    		$$invalidate(0, value);
    	}

    	function textarea_1_input_handler() {
    		value = this.value;
    		$$invalidate(0, value);
    	}

    	const click_handler_3 = () => dispatch("click-append");
    	const click_handler_4 = () => dispatch("click-prepend");

    	$$self.$$set = $$new_props => {
    		$$invalidate(77, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("outlined" in $$new_props) $$invalidate(2, outlined = $$new_props.outlined);
    		if ("value" in $$new_props) $$invalidate(0, value = $$new_props.value);
    		if ("label" in $$new_props) $$invalidate(3, label = $$new_props.label);
    		if ("placeholder" in $$new_props) $$invalidate(4, placeholder = $$new_props.placeholder);
    		if ("hint" in $$new_props) $$invalidate(5, hint = $$new_props.hint);
    		if ("error" in $$new_props) $$invalidate(6, error = $$new_props.error);
    		if ("append" in $$new_props) $$invalidate(7, append = $$new_props.append);
    		if ("prepend" in $$new_props) $$invalidate(8, prepend = $$new_props.prepend);
    		if ("persistentHint" in $$new_props) $$invalidate(31, persistentHint = $$new_props.persistentHint);
    		if ("textarea" in $$new_props) $$invalidate(9, textarea = $$new_props.textarea);
    		if ("rows" in $$new_props) $$invalidate(10, rows = $$new_props.rows);
    		if ("select" in $$new_props) $$invalidate(11, select = $$new_props.select);
    		if ("dense" in $$new_props) $$invalidate(12, dense = $$new_props.dense);
    		if ("autocomplete" in $$new_props) $$invalidate(13, autocomplete = $$new_props.autocomplete);
    		if ("noUnderline" in $$new_props) $$invalidate(14, noUnderline = $$new_props.noUnderline);
    		if ("appendReverse" in $$new_props) $$invalidate(15, appendReverse = $$new_props.appendReverse);
    		if ("prependReverse" in $$new_props) $$invalidate(16, prependReverse = $$new_props.prependReverse);
    		if ("color" in $$new_props) $$invalidate(17, color = $$new_props.color);
    		if ("bgColor" in $$new_props) $$invalidate(18, bgColor = $$new_props.bgColor);
    		if ("iconClass" in $$new_props) $$invalidate(19, iconClass = $$new_props.iconClass);
    		if ("disabled" in $$new_props) $$invalidate(20, disabled = $$new_props.disabled);
    		if ("add" in $$new_props) $$invalidate(32, add = $$new_props.add);
    		if ("remove" in $$new_props) $$invalidate(33, remove = $$new_props.remove);
    		if ("replace" in $$new_props) $$invalidate(34, replace = $$new_props.replace);
    		if ("inputClasses" in $$new_props) $$invalidate(35, inputClasses = $$new_props.inputClasses);
    		if ("classes" in $$new_props) $$invalidate(36, classes = $$new_props.classes);
    		if ("appendClasses" in $$new_props) $$invalidate(37, appendClasses = $$new_props.appendClasses);
    		if ("prependClasses" in $$new_props) $$invalidate(38, prependClasses = $$new_props.prependClasses);
    		if ("extend" in $$new_props) $$invalidate(39, extend = $$new_props.extend);
    		if ("focused" in $$new_props) $$invalidate(1, focused = $$new_props.focused);
    		if ("$$scope" in $$new_props) $$invalidate(69, $$scope = $$new_props.$$scope);
    	};

    	$$self.$capture_state = () => ({
    		createEventDispatcher,
    		utils,
    		ClassBuilder,
    		filterProps,
    		Icon,
    		Label,
    		Hint,
    		Underline,
    		outlined,
    		value,
    		label,
    		placeholder,
    		hint,
    		error,
    		append,
    		prepend,
    		persistentHint,
    		textarea,
    		rows,
    		select,
    		dense,
    		autocomplete,
    		noUnderline,
    		appendReverse,
    		prependReverse,
    		color,
    		bgColor,
    		iconClass,
    		disabled,
    		inputDefault,
    		classesDefault: classesDefault$1,
    		appendDefault,
    		prependDefault,
    		add,
    		remove,
    		replace,
    		inputClasses,
    		classes,
    		appendClasses,
    		prependClasses,
    		bg,
    		border,
    		txt,
    		caret,
    		cb,
    		ccb,
    		acb,
    		pcb,
    		extend,
    		focused,
    		wClasses,
    		aClasses,
    		pClasses,
    		toggleFocused,
    		props,
    		dispatch,
    		showHint,
    		labelOnTop,
    		iClasses
    	});

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(77, $$props = assign(assign({}, $$props), $$new_props));
    		if ("outlined" in $$props) $$invalidate(2, outlined = $$new_props.outlined);
    		if ("value" in $$props) $$invalidate(0, value = $$new_props.value);
    		if ("label" in $$props) $$invalidate(3, label = $$new_props.label);
    		if ("placeholder" in $$props) $$invalidate(4, placeholder = $$new_props.placeholder);
    		if ("hint" in $$props) $$invalidate(5, hint = $$new_props.hint);
    		if ("error" in $$props) $$invalidate(6, error = $$new_props.error);
    		if ("append" in $$props) $$invalidate(7, append = $$new_props.append);
    		if ("prepend" in $$props) $$invalidate(8, prepend = $$new_props.prepend);
    		if ("persistentHint" in $$props) $$invalidate(31, persistentHint = $$new_props.persistentHint);
    		if ("textarea" in $$props) $$invalidate(9, textarea = $$new_props.textarea);
    		if ("rows" in $$props) $$invalidate(10, rows = $$new_props.rows);
    		if ("select" in $$props) $$invalidate(11, select = $$new_props.select);
    		if ("dense" in $$props) $$invalidate(12, dense = $$new_props.dense);
    		if ("autocomplete" in $$props) $$invalidate(13, autocomplete = $$new_props.autocomplete);
    		if ("noUnderline" in $$props) $$invalidate(14, noUnderline = $$new_props.noUnderline);
    		if ("appendReverse" in $$props) $$invalidate(15, appendReverse = $$new_props.appendReverse);
    		if ("prependReverse" in $$props) $$invalidate(16, prependReverse = $$new_props.prependReverse);
    		if ("color" in $$props) $$invalidate(17, color = $$new_props.color);
    		if ("bgColor" in $$props) $$invalidate(18, bgColor = $$new_props.bgColor);
    		if ("iconClass" in $$props) $$invalidate(19, iconClass = $$new_props.iconClass);
    		if ("disabled" in $$props) $$invalidate(20, disabled = $$new_props.disabled);
    		if ("add" in $$props) $$invalidate(32, add = $$new_props.add);
    		if ("remove" in $$props) $$invalidate(33, remove = $$new_props.remove);
    		if ("replace" in $$props) $$invalidate(34, replace = $$new_props.replace);
    		if ("inputClasses" in $$props) $$invalidate(35, inputClasses = $$new_props.inputClasses);
    		if ("classes" in $$props) $$invalidate(36, classes = $$new_props.classes);
    		if ("appendClasses" in $$props) $$invalidate(37, appendClasses = $$new_props.appendClasses);
    		if ("prependClasses" in $$props) $$invalidate(38, prependClasses = $$new_props.prependClasses);
    		if ("extend" in $$props) $$invalidate(39, extend = $$new_props.extend);
    		if ("focused" in $$props) $$invalidate(1, focused = $$new_props.focused);
    		if ("wClasses" in $$props) $$invalidate(21, wClasses = $$new_props.wClasses);
    		if ("aClasses" in $$props) $$invalidate(22, aClasses = $$new_props.aClasses);
    		if ("pClasses" in $$props) $$invalidate(23, pClasses = $$new_props.pClasses);
    		if ("showHint" in $$props) $$invalidate(24, showHint = $$new_props.showHint);
    		if ("labelOnTop" in $$props) $$invalidate(25, labelOnTop = $$new_props.labelOnTop);
    		if ("iClasses" in $$props) $$invalidate(26, iClasses = $$new_props.iClasses);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		if ($$self.$$.dirty[0] & /*error, hint, focused*/ 98 | $$self.$$.dirty[1] & /*persistentHint*/ 1) {
    			$$invalidate(24, showHint = error || (persistentHint ? hint : focused && hint));
    		}

    		if ($$self.$$.dirty[0] & /*placeholder, focused, value*/ 19) {
    			$$invalidate(25, labelOnTop = placeholder || focused || (value || value === 0));
    		}

    		$$invalidate(26, iClasses = cb.flush().remove("pt-6 pb-2", outlined).add("border rounded bg-transparent py-4 duration-200 ease-in", outlined).add("border-error-500 caret-error-500", error).remove(caret(), error).add(caret(), !error).add(border(), outlined && focused && !error).add("bg-gray-100 dark:bg-dark-600", !outlined).add("bg-gray-300 dark:bg-dark-200", focused && !outlined).remove("px-4", prepend).add("pr-4 pl-10", prepend).add(add).remove("pt-6 pb-2", dense && !outlined).add("pt-4 pb-1", dense && !outlined).remove("bg-gray-100", disabled).add("bg-gray-50", disabled).add("cursor-pointer", select && !autocomplete).add($$props.class).remove(remove).replace(replace).extend(extend).get());

    		if ($$self.$$.dirty[0] & /*select, autocomplete, dense, outlined, error, disabled*/ 1062980) {
    			$$invalidate(21, wClasses = ccb.flush().add("select", select || autocomplete).add("dense", dense && !outlined).remove("mb-6 mt-2", dense && !outlined).add("mb-4 mt-1", dense).replace({ "text-gray-600": "text-error-500" }, error).add("text-gray-200", disabled).get());
    		}
    	};

    	$$invalidate(22, aClasses = acb.flush().get());
    	$$invalidate(23, pClasses = pcb.flush().get());
    	$$props = exclude_internal_props($$props);

    	return [
    		value,
    		focused,
    		outlined,
    		label,
    		placeholder,
    		hint,
    		error,
    		append,
    		prepend,
    		textarea,
    		rows,
    		select,
    		dense,
    		autocomplete,
    		noUnderline,
    		appendReverse,
    		prependReverse,
    		color,
    		bgColor,
    		iconClass,
    		disabled,
    		wClasses,
    		aClasses,
    		pClasses,
    		showHint,
    		labelOnTop,
    		iClasses,
    		txt,
    		toggleFocused,
    		props,
    		dispatch,
    		persistentHint,
    		add,
    		remove,
    		replace,
    		inputClasses,
    		classes,
    		appendClasses,
    		prependClasses,
    		extend,
    		slots,
    		blur_handler,
    		change_handler,
    		input_handler,
    		keydown_handler,
    		keypress_handler,
    		keyup_handler,
    		click_handler,
    		focus_handler,
    		change_handler_1,
    		input_handler_1,
    		keydown_handler_1,
    		keypress_handler_1,
    		keyup_handler_1,
    		click_handler_1,
    		focus_handler_1,
    		blur_handler_1,
    		change_handler_2,
    		input_handler_2,
    		keydown_handler_2,
    		keypress_handler_2,
    		keyup_handler_2,
    		click_handler_2,
    		blur_handler_2,
    		focus_handler_2,
    		input_input_handler,
    		textarea_1_input_handler,
    		click_handler_3,
    		click_handler_4,
    		$$scope
    	];
    }

    class TextField extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(
    			this,
    			options,
    			instance$7,
    			create_fragment$7,
    			safe_not_equal,
    			{
    				outlined: 2,
    				value: 0,
    				label: 3,
    				placeholder: 4,
    				hint: 5,
    				error: 6,
    				append: 7,
    				prepend: 8,
    				persistentHint: 31,
    				textarea: 9,
    				rows: 10,
    				select: 11,
    				dense: 12,
    				autocomplete: 13,
    				noUnderline: 14,
    				appendReverse: 15,
    				prependReverse: 16,
    				color: 17,
    				bgColor: 18,
    				iconClass: 19,
    				disabled: 20,
    				add: 32,
    				remove: 33,
    				replace: 34,
    				inputClasses: 35,
    				classes: 36,
    				appendClasses: 37,
    				prependClasses: 38,
    				extend: 39,
    				focused: 1
    			},
    			[-1, -1, -1]
    		);

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "TextField",
    			options,
    			id: create_fragment$7.name
    		});
    	}

    	get outlined() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set outlined(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get value() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set value(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get label() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set label(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get placeholder() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set placeholder(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get hint() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set hint(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get error() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set error(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get append() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set append(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get prepend() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set prepend(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get persistentHint() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set persistentHint(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get textarea() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set textarea(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get rows() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set rows(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get select() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set select(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get dense() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set dense(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get autocomplete() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set autocomplete(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get noUnderline() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set noUnderline(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get appendReverse() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set appendReverse(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get prependReverse() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set prependReverse(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get color() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set color(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get bgColor() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set bgColor(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get iconClass() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set iconClass(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get disabled() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set disabled(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get add() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set add(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get remove() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set remove(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get replace() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set replace(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get inputClasses() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set inputClasses(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get classes() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set classes(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get appendClasses() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set appendClasses(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get prependClasses() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set prependClasses(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get extend() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set extend(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get focused() {
    		throw new Error("<TextField>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set focused(value) {
    		throw new Error("<TextField>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    const subscriber_queue = [];
    /**
     * Create a `Writable` store that allows both updating and reading by subscription.
     * @param {*=}value initial value
     * @param {StartStopNotifier=}start start and stop notifications for subscriptions
     */
    function writable(value, start = noop) {
        let stop;
        const subscribers = [];
        function set(new_value) {
            if (safe_not_equal(value, new_value)) {
                value = new_value;
                if (stop) { // store is ready
                    const run_queue = !subscriber_queue.length;
                    for (let i = 0; i < subscribers.length; i += 1) {
                        const s = subscribers[i];
                        s[1]();
                        subscriber_queue.push(s, value);
                    }
                    if (run_queue) {
                        for (let i = 0; i < subscriber_queue.length; i += 2) {
                            subscriber_queue[i][0](subscriber_queue[i + 1]);
                        }
                        subscriber_queue.length = 0;
                    }
                }
            }
        }
        function update(fn) {
            set(fn(value));
        }
        function subscribe(run, invalidate = noop) {
            const subscriber = [run, invalidate];
            subscribers.push(subscriber);
            if (subscribers.length === 1) {
                stop = start(set) || noop;
            }
            run(value);
            return () => {
                const index = subscribers.indexOf(subscriber);
                if (index !== -1) {
                    subscribers.splice(index, 1);
                }
                if (subscribers.length === 0) {
                    stop();
                    stop = null;
                }
            };
        }
        return { set, update, subscribe };
    }

    /* node_modules\smelte\src\components\Ripple\Ripple.svelte generated by Svelte v3.38.2 */
    const file$5 = "node_modules\\smelte\\src\\components\\Ripple\\Ripple.svelte";

    function create_fragment$6(ctx) {
    	let span;
    	let span_class_value;
    	let current;
    	let mounted;
    	let dispose;
    	const default_slot_template = /*#slots*/ ctx[6].default;
    	const default_slot = create_slot(default_slot_template, ctx, /*$$scope*/ ctx[5], null);

    	const block = {
    		c: function create() {
    			span = element("span");
    			if (default_slot) default_slot.c();
    			attr_dev(span, "class", span_class_value = "z-40 " + /*$$props*/ ctx[3].class + " p-2 rounded-full flex items-center justify-center top-0 left-0 " + (/*noHover*/ ctx[0] ? "" : /*hoverClass*/ ctx[2]) + " svelte-1o8z87d");
    			add_location(span, file$5, 15, 0, 293);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, span, anchor);

    			if (default_slot) {
    				default_slot.m(span, null);
    			}

    			current = true;

    			if (!mounted) {
    				dispose = action_destroyer(/*ripple*/ ctx[1].call(null, span));
    				mounted = true;
    			}
    		},
    		p: function update(ctx, [dirty]) {
    			if (default_slot) {
    				if (default_slot.p && (!current || dirty & /*$$scope*/ 32)) {
    					update_slot(default_slot, default_slot_template, ctx, /*$$scope*/ ctx[5], dirty, null, null);
    				}
    			}

    			if (!current || dirty & /*$$props, noHover, hoverClass*/ 13 && span_class_value !== (span_class_value = "z-40 " + /*$$props*/ ctx[3].class + " p-2 rounded-full flex items-center justify-center top-0 left-0 " + (/*noHover*/ ctx[0] ? "" : /*hoverClass*/ ctx[2]) + " svelte-1o8z87d")) {
    				attr_dev(span, "class", span_class_value);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(default_slot, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(default_slot, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(span);
    			if (default_slot) default_slot.d(detaching);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$6.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$6($$self, $$props, $$invalidate) {
    	let ripple;
    	let hoverClass;
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("Ripple", slots, ['default']);
    	let { color = "primary" } = $$props;
    	let { noHover = false } = $$props;

    	$$self.$$set = $$new_props => {
    		$$invalidate(3, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("color" in $$new_props) $$invalidate(4, color = $$new_props.color);
    		if ("noHover" in $$new_props) $$invalidate(0, noHover = $$new_props.noHover);
    		if ("$$scope" in $$new_props) $$invalidate(5, $$scope = $$new_props.$$scope);
    	};

    	$$self.$capture_state = () => ({
    		color,
    		noHover,
    		createRipple: r,
    		ripple,
    		hoverClass
    	});

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(3, $$props = assign(assign({}, $$props), $$new_props));
    		if ("color" in $$props) $$invalidate(4, color = $$new_props.color);
    		if ("noHover" in $$props) $$invalidate(0, noHover = $$new_props.noHover);
    		if ("ripple" in $$props) $$invalidate(1, ripple = $$new_props.ripple);
    		if ("hoverClass" in $$props) $$invalidate(2, hoverClass = $$new_props.hoverClass);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		if ($$self.$$.dirty & /*color*/ 16) {
    			$$invalidate(1, ripple = r(color, true));
    		}

    		if ($$self.$$.dirty & /*color*/ 16) {
    			$$invalidate(2, hoverClass = `hover:bg-${color}-transLight`);
    		}
    	};

    	$$props = exclude_internal_props($$props);
    	return [noHover, ripple, hoverClass, $$props, color, $$scope, slots];
    }

    class Ripple extends SvelteComponentDev {
    	constructor(options) {
    		super(options);
    		init(this, options, instance$6, create_fragment$6, safe_not_equal, { color: 4, noHover: 0 });

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Ripple",
    			options,
    			id: create_fragment$6.name
    		});
    	}

    	get color() {
    		throw new Error("<Ripple>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set color(value) {
    		throw new Error("<Ripple>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get noHover() {
    		throw new Error("<Ripple>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set noHover(value) {
    		throw new Error("<Ripple>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    function hideListAction(node, cb) {
      const onWindowClick = e => {
        if (!node.contains(e.target)) {
          cb();
        }
      };

      window.addEventListener("click", onWindowClick);

      return {
        destroy: () => {
          window.removeEventListener("click", onWindowClick);
        }
      };
    }

    /* node_modules\smelte\src\components\Select\Select.svelte generated by Svelte v3.38.2 */
    const file$4 = "node_modules\\smelte\\src\\components\\Select\\Select.svelte";
    const get_options_slot_changes = dirty => ({});
    const get_options_slot_context = ctx => ({});
    const get_select_slot_changes = dirty => ({});
    const get_select_slot_context = ctx => ({});

    // (114:22)      
    function fallback_block_1(ctx) {
    	let textfield;
    	let current;

    	textfield = new TextField({
    			props: {
    				select: true,
    				dense: /*dense*/ ctx[10],
    				focused: /*showList*/ ctx[1],
    				autocomplete: /*autocomplete*/ ctx[12],
    				value: /*selectedLabel*/ ctx[24],
    				outlined: /*outlined*/ ctx[5],
    				label: /*label*/ ctx[3],
    				placeholder: /*placeholder*/ ctx[6],
    				hint: /*hint*/ ctx[7],
    				error: /*error*/ ctx[8],
    				append: /*append*/ ctx[9],
    				persistentHint: /*persistentHint*/ ctx[11],
    				color: /*color*/ ctx[4],
    				add: /*add*/ ctx[21],
    				remove: /*remove*/ ctx[22],
    				replace: /*replace*/ ctx[23],
    				noUnderline: /*noUnderline*/ ctx[13],
    				class: /*inputWrapperClasses*/ ctx[14],
    				appendClasses: /*appendClasses*/ ctx[2],
    				labelClasses: /*labelClasses*/ ctx[15],
    				inputClasses: /*inputClasses*/ ctx[16],
    				prependClasses: /*prependClasses*/ ctx[17],
    				appendReverse: /*showList*/ ctx[1]
    			},
    			$$inline: true
    		});

    	textfield.$on("click", /*handleInputClick*/ ctx[30]);
    	textfield.$on("click-append", /*click_append_handler*/ ctx[41]);
    	textfield.$on("click", /*click_handler*/ ctx[42]);
    	textfield.$on("input", /*filterItems*/ ctx[29]);

    	const block = {
    		c: function create() {
    			create_component(textfield.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(textfield, target, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const textfield_changes = {};
    			if (dirty[0] & /*dense*/ 1024) textfield_changes.dense = /*dense*/ ctx[10];
    			if (dirty[0] & /*showList*/ 2) textfield_changes.focused = /*showList*/ ctx[1];
    			if (dirty[0] & /*autocomplete*/ 4096) textfield_changes.autocomplete = /*autocomplete*/ ctx[12];
    			if (dirty[0] & /*selectedLabel*/ 16777216) textfield_changes.value = /*selectedLabel*/ ctx[24];
    			if (dirty[0] & /*outlined*/ 32) textfield_changes.outlined = /*outlined*/ ctx[5];
    			if (dirty[0] & /*label*/ 8) textfield_changes.label = /*label*/ ctx[3];
    			if (dirty[0] & /*placeholder*/ 64) textfield_changes.placeholder = /*placeholder*/ ctx[6];
    			if (dirty[0] & /*hint*/ 128) textfield_changes.hint = /*hint*/ ctx[7];
    			if (dirty[0] & /*error*/ 256) textfield_changes.error = /*error*/ ctx[8];
    			if (dirty[0] & /*append*/ 512) textfield_changes.append = /*append*/ ctx[9];
    			if (dirty[0] & /*persistentHint*/ 2048) textfield_changes.persistentHint = /*persistentHint*/ ctx[11];
    			if (dirty[0] & /*color*/ 16) textfield_changes.color = /*color*/ ctx[4];
    			if (dirty[0] & /*add*/ 2097152) textfield_changes.add = /*add*/ ctx[21];
    			if (dirty[0] & /*remove*/ 4194304) textfield_changes.remove = /*remove*/ ctx[22];
    			if (dirty[0] & /*replace*/ 8388608) textfield_changes.replace = /*replace*/ ctx[23];
    			if (dirty[0] & /*noUnderline*/ 8192) textfield_changes.noUnderline = /*noUnderline*/ ctx[13];
    			if (dirty[0] & /*inputWrapperClasses*/ 16384) textfield_changes.class = /*inputWrapperClasses*/ ctx[14];
    			if (dirty[0] & /*appendClasses*/ 4) textfield_changes.appendClasses = /*appendClasses*/ ctx[2];
    			if (dirty[0] & /*labelClasses*/ 32768) textfield_changes.labelClasses = /*labelClasses*/ ctx[15];
    			if (dirty[0] & /*inputClasses*/ 65536) textfield_changes.inputClasses = /*inputClasses*/ ctx[16];
    			if (dirty[0] & /*prependClasses*/ 131072) textfield_changes.prependClasses = /*prependClasses*/ ctx[17];
    			if (dirty[0] & /*showList*/ 2) textfield_changes.appendReverse = /*showList*/ ctx[1];
    			textfield.$set(textfield_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(textfield.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(textfield.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(textfield, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: fallback_block_1.name,
    		type: "fallback",
    		source: "(114:22)      ",
    		ctx
    	});

    	return block;
    }

    // (146:2) {#if showList}
    function create_if_block$1(ctx) {
    	let current;
    	const options_slot_template = /*#slots*/ ctx[40].options;
    	const options_slot = create_slot(options_slot_template, ctx, /*$$scope*/ ctx[39], get_options_slot_context);
    	const options_slot_or_fallback = options_slot || fallback_block(ctx);

    	const block = {
    		c: function create() {
    			if (options_slot_or_fallback) options_slot_or_fallback.c();
    		},
    		m: function mount(target, anchor) {
    			if (options_slot_or_fallback) {
    				options_slot_or_fallback.m(target, anchor);
    			}

    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			if (options_slot) {
    				if (options_slot.p && (!current || dirty[1] & /*$$scope*/ 256)) {
    					update_slot(options_slot, options_slot_template, ctx, /*$$scope*/ ctx[39], dirty, get_options_slot_changes, get_options_slot_context);
    				}
    			} else {
    				if (options_slot_or_fallback && options_slot_or_fallback.p && dirty[0] & /*o, showList, listClasses, selectedClasses, itemClasses, dense, filteredItems, value*/ 169608195) {
    					options_slot_or_fallback.p(ctx, dirty);
    				}
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(options_slot_or_fallback, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(options_slot_or_fallback, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (options_slot_or_fallback) options_slot_or_fallback.d(detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block$1.name,
    		type: "if",
    		source: "(146:2) {#if showList}",
    		ctx
    	});

    	return block;
    }

    // (147:25)        
    function fallback_block(ctx) {
    	let div;
    	let list;
    	let updating_value;
    	let current;
    	let mounted;
    	let dispose;

    	function list_value_binding(value) {
    		/*list_value_binding*/ ctx[43](value);
    	}

    	let list_props = {
    		class: /*listClasses*/ ctx[18],
    		selectedClasses: /*selectedClasses*/ ctx[19],
    		itemClasses: /*itemClasses*/ ctx[20],
    		select: true,
    		dense: /*dense*/ ctx[10],
    		items: /*filteredItems*/ ctx[25]
    	};

    	if (/*value*/ ctx[0] !== void 0) {
    		list_props.value = /*value*/ ctx[0];
    	}

    	list = new List({ props: list_props, $$inline: true });
    	binding_callbacks.push(() => bind(list, "value", list_value_binding));
    	list.$on("change", /*change_handler*/ ctx[44]);

    	const block = {
    		c: function create() {
    			div = element("div");
    			create_component(list.$$.fragment);
    			attr_dev(div, "class", /*o*/ ctx[27]);
    			add_location(div, file$4, 147, 6, 3663);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div, anchor);
    			mount_component(list, div, null);
    			current = true;

    			if (!mounted) {
    				dispose = listen_dev(div, "click", /*click_handler_1*/ ctx[45], false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			const list_changes = {};
    			if (dirty[0] & /*listClasses*/ 262144) list_changes.class = /*listClasses*/ ctx[18];
    			if (dirty[0] & /*selectedClasses*/ 524288) list_changes.selectedClasses = /*selectedClasses*/ ctx[19];
    			if (dirty[0] & /*itemClasses*/ 1048576) list_changes.itemClasses = /*itemClasses*/ ctx[20];
    			if (dirty[0] & /*dense*/ 1024) list_changes.dense = /*dense*/ ctx[10];
    			if (dirty[0] & /*filteredItems*/ 33554432) list_changes.items = /*filteredItems*/ ctx[25];

    			if (!updating_value && dirty[0] & /*value*/ 1) {
    				updating_value = true;
    				list_changes.value = /*value*/ ctx[0];
    				add_flush_callback(() => updating_value = false);
    			}

    			list.$set(list_changes);

    			if (!current || dirty[0] & /*o*/ 134217728) {
    				attr_dev(div, "class", /*o*/ ctx[27]);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(list.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(list.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div);
    			destroy_component(list);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: fallback_block.name,
    		type: "fallback",
    		source: "(147:25)        ",
    		ctx
    	});

    	return block;
    }

    function create_fragment$5(ctx) {
    	let div;
    	let t;
    	let current;
    	let mounted;
    	let dispose;
    	const select_slot_template = /*#slots*/ ctx[40].select;
    	const select_slot = create_slot(select_slot_template, ctx, /*$$scope*/ ctx[39], get_select_slot_context);
    	const select_slot_or_fallback = select_slot || fallback_block_1(ctx);
    	let if_block = /*showList*/ ctx[1] && create_if_block$1(ctx);

    	const block = {
    		c: function create() {
    			div = element("div");
    			if (select_slot_or_fallback) select_slot_or_fallback.c();
    			t = space();
    			if (if_block) if_block.c();
    			attr_dev(div, "class", /*c*/ ctx[26]);
    			add_location(div, file$4, 112, 0, 2929);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div, anchor);

    			if (select_slot_or_fallback) {
    				select_slot_or_fallback.m(div, null);
    			}

    			append_dev(div, t);
    			if (if_block) if_block.m(div, null);
    			current = true;

    			if (!mounted) {
    				dispose = action_destroyer(hideListAction.call(null, div, /*onHideListPanel*/ ctx[31]));
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (select_slot) {
    				if (select_slot.p && (!current || dirty[1] & /*$$scope*/ 256)) {
    					update_slot(select_slot, select_slot_template, ctx, /*$$scope*/ ctx[39], dirty, get_select_slot_changes, get_select_slot_context);
    				}
    			} else {
    				if (select_slot_or_fallback && select_slot_or_fallback.p && dirty[0] & /*dense, showList, autocomplete, selectedLabel, outlined, label, placeholder, hint, error, append, persistentHint, color, add, remove, replace, noUnderline, inputWrapperClasses, appendClasses, labelClasses, inputClasses, prependClasses*/ 31719422) {
    					select_slot_or_fallback.p(ctx, dirty);
    				}
    			}

    			if (/*showList*/ ctx[1]) {
    				if (if_block) {
    					if_block.p(ctx, dirty);

    					if (dirty[0] & /*showList*/ 2) {
    						transition_in(if_block, 1);
    					}
    				} else {
    					if_block = create_if_block$1(ctx);
    					if_block.c();
    					transition_in(if_block, 1);
    					if_block.m(div, null);
    				}
    			} else if (if_block) {
    				group_outros();

    				transition_out(if_block, 1, 1, () => {
    					if_block = null;
    				});

    				check_outros();
    			}

    			if (!current || dirty[0] & /*c*/ 67108864) {
    				attr_dev(div, "class", /*c*/ ctx[26]);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(select_slot_or_fallback, local);
    			transition_in(if_block);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(select_slot_or_fallback, local);
    			transition_out(if_block);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div);
    			if (select_slot_or_fallback) select_slot_or_fallback.d(detaching);
    			if (if_block) if_block.d();
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$5.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    const optionsClassesDefault = "absolute left-0 bg-white rounded shadow w-full z-20 dark:bg-dark-500";
    const classesDefault = "cursor-pointer relative pb-4";

    function process(it) {
    	return it.map(i => typeof i !== "object" ? { value: i, text: i } : i);
    }

    function instance$5($$self, $$props, $$invalidate) {
    	let filteredItems;
    	let c;
    	let o;
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("Select", slots, ['select','options']);
    	const noop = i => i;
    	let { items = [] } = $$props;
    	let { value = "" } = $$props;
    	const text = "";
    	let { label = "" } = $$props;
    	let { selectedLabel: selectedLabelProp = undefined } = $$props;
    	let { color = "primary" } = $$props;
    	let { outlined = false } = $$props;
    	let { placeholder = "" } = $$props;
    	let { hint = "" } = $$props;
    	let { error = false } = $$props;
    	let { append = "arrow_drop_down" } = $$props;
    	let { dense = false } = $$props;
    	let { persistentHint = false } = $$props;
    	let { autocomplete = false } = $$props;
    	let { noUnderline = false } = $$props;
    	let { showList = false } = $$props;
    	let { classes = classesDefault } = $$props;
    	let { optionsClasses = optionsClassesDefault } = $$props;
    	let { inputWrapperClasses = noop } = $$props;
    	let { appendClasses = noop } = $$props;
    	let { labelClasses = noop } = $$props;
    	let { inputClasses = noop } = $$props;
    	let { prependClasses = noop } = $$props;
    	let { listClasses = noop } = $$props;
    	let { selectedClasses = noop } = $$props;
    	let { itemClasses = noop } = $$props;
    	let { add = "" } = $$props;
    	let { remove = "" } = $$props;
    	let { replace = "" } = $$props;
    	let itemsProcessed = [];
    	const dispatch = createEventDispatcher();
    	let selectedLabel = "";
    	let filterText = null;

    	function filterItems({ target }) {
    		$$invalidate(38, filterText = target.value.toLowerCase());
    	}

    	function handleInputClick() {
    		if (autocomplete) {
    			$$invalidate(1, showList = true);
    		} else {
    			$$invalidate(1, showList = !showList);
    		}
    	}

    	const onHideListPanel = () => $$invalidate(1, showList = false);
    	const cb = new ClassBuilder(classes, classesDefault);
    	const ocb = new ClassBuilder(optionsClasses, optionsClassesDefault);
    	const click_append_handler = e => $$invalidate(1, showList = !showList);

    	function click_handler(event) {
    		bubble($$self, event);
    	}

    	function list_value_binding(value$1) {
    		value = value$1;
    		$$invalidate(0, value);
    	}

    	const change_handler = ({ detail }) => {
    		dispatch("change", detail);
    	};

    	const click_handler_1 = () => $$invalidate(1, showList = false);

    	$$self.$$set = $$new_props => {
    		$$invalidate(49, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("items" in $$new_props) $$invalidate(32, items = $$new_props.items);
    		if ("value" in $$new_props) $$invalidate(0, value = $$new_props.value);
    		if ("label" in $$new_props) $$invalidate(3, label = $$new_props.label);
    		if ("selectedLabel" in $$new_props) $$invalidate(34, selectedLabelProp = $$new_props.selectedLabel);
    		if ("color" in $$new_props) $$invalidate(4, color = $$new_props.color);
    		if ("outlined" in $$new_props) $$invalidate(5, outlined = $$new_props.outlined);
    		if ("placeholder" in $$new_props) $$invalidate(6, placeholder = $$new_props.placeholder);
    		if ("hint" in $$new_props) $$invalidate(7, hint = $$new_props.hint);
    		if ("error" in $$new_props) $$invalidate(8, error = $$new_props.error);
    		if ("append" in $$new_props) $$invalidate(9, append = $$new_props.append);
    		if ("dense" in $$new_props) $$invalidate(10, dense = $$new_props.dense);
    		if ("persistentHint" in $$new_props) $$invalidate(11, persistentHint = $$new_props.persistentHint);
    		if ("autocomplete" in $$new_props) $$invalidate(12, autocomplete = $$new_props.autocomplete);
    		if ("noUnderline" in $$new_props) $$invalidate(13, noUnderline = $$new_props.noUnderline);
    		if ("showList" in $$new_props) $$invalidate(1, showList = $$new_props.showList);
    		if ("classes" in $$new_props) $$invalidate(35, classes = $$new_props.classes);
    		if ("optionsClasses" in $$new_props) $$invalidate(36, optionsClasses = $$new_props.optionsClasses);
    		if ("inputWrapperClasses" in $$new_props) $$invalidate(14, inputWrapperClasses = $$new_props.inputWrapperClasses);
    		if ("appendClasses" in $$new_props) $$invalidate(2, appendClasses = $$new_props.appendClasses);
    		if ("labelClasses" in $$new_props) $$invalidate(15, labelClasses = $$new_props.labelClasses);
    		if ("inputClasses" in $$new_props) $$invalidate(16, inputClasses = $$new_props.inputClasses);
    		if ("prependClasses" in $$new_props) $$invalidate(17, prependClasses = $$new_props.prependClasses);
    		if ("listClasses" in $$new_props) $$invalidate(18, listClasses = $$new_props.listClasses);
    		if ("selectedClasses" in $$new_props) $$invalidate(19, selectedClasses = $$new_props.selectedClasses);
    		if ("itemClasses" in $$new_props) $$invalidate(20, itemClasses = $$new_props.itemClasses);
    		if ("add" in $$new_props) $$invalidate(21, add = $$new_props.add);
    		if ("remove" in $$new_props) $$invalidate(22, remove = $$new_props.remove);
    		if ("replace" in $$new_props) $$invalidate(23, replace = $$new_props.replace);
    		if ("$$scope" in $$new_props) $$invalidate(39, $$scope = $$new_props.$$scope);
    	};

    	$$self.$capture_state = () => ({
    		createEventDispatcher,
    		onMount,
    		quadOut,
    		quadIn,
    		List,
    		TextField,
    		ClassBuilder,
    		hideListAction,
    		optionsClassesDefault,
    		classesDefault,
    		noop,
    		items,
    		value,
    		text,
    		label,
    		selectedLabelProp,
    		color,
    		outlined,
    		placeholder,
    		hint,
    		error,
    		append,
    		dense,
    		persistentHint,
    		autocomplete,
    		noUnderline,
    		showList,
    		classes,
    		optionsClasses,
    		inputWrapperClasses,
    		appendClasses,
    		labelClasses,
    		inputClasses,
    		prependClasses,
    		listClasses,
    		selectedClasses,
    		itemClasses,
    		add,
    		remove,
    		replace,
    		itemsProcessed,
    		process,
    		dispatch,
    		selectedLabel,
    		filterText,
    		filterItems,
    		handleInputClick,
    		onHideListPanel,
    		cb,
    		ocb,
    		filteredItems,
    		c,
    		o
    	});

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(49, $$props = assign(assign({}, $$props), $$new_props));
    		if ("items" in $$props) $$invalidate(32, items = $$new_props.items);
    		if ("value" in $$props) $$invalidate(0, value = $$new_props.value);
    		if ("label" in $$props) $$invalidate(3, label = $$new_props.label);
    		if ("selectedLabelProp" in $$props) $$invalidate(34, selectedLabelProp = $$new_props.selectedLabelProp);
    		if ("color" in $$props) $$invalidate(4, color = $$new_props.color);
    		if ("outlined" in $$props) $$invalidate(5, outlined = $$new_props.outlined);
    		if ("placeholder" in $$props) $$invalidate(6, placeholder = $$new_props.placeholder);
    		if ("hint" in $$props) $$invalidate(7, hint = $$new_props.hint);
    		if ("error" in $$props) $$invalidate(8, error = $$new_props.error);
    		if ("append" in $$props) $$invalidate(9, append = $$new_props.append);
    		if ("dense" in $$props) $$invalidate(10, dense = $$new_props.dense);
    		if ("persistentHint" in $$props) $$invalidate(11, persistentHint = $$new_props.persistentHint);
    		if ("autocomplete" in $$props) $$invalidate(12, autocomplete = $$new_props.autocomplete);
    		if ("noUnderline" in $$props) $$invalidate(13, noUnderline = $$new_props.noUnderline);
    		if ("showList" in $$props) $$invalidate(1, showList = $$new_props.showList);
    		if ("classes" in $$props) $$invalidate(35, classes = $$new_props.classes);
    		if ("optionsClasses" in $$props) $$invalidate(36, optionsClasses = $$new_props.optionsClasses);
    		if ("inputWrapperClasses" in $$props) $$invalidate(14, inputWrapperClasses = $$new_props.inputWrapperClasses);
    		if ("appendClasses" in $$props) $$invalidate(2, appendClasses = $$new_props.appendClasses);
    		if ("labelClasses" in $$props) $$invalidate(15, labelClasses = $$new_props.labelClasses);
    		if ("inputClasses" in $$props) $$invalidate(16, inputClasses = $$new_props.inputClasses);
    		if ("prependClasses" in $$props) $$invalidate(17, prependClasses = $$new_props.prependClasses);
    		if ("listClasses" in $$props) $$invalidate(18, listClasses = $$new_props.listClasses);
    		if ("selectedClasses" in $$props) $$invalidate(19, selectedClasses = $$new_props.selectedClasses);
    		if ("itemClasses" in $$props) $$invalidate(20, itemClasses = $$new_props.itemClasses);
    		if ("add" in $$props) $$invalidate(21, add = $$new_props.add);
    		if ("remove" in $$props) $$invalidate(22, remove = $$new_props.remove);
    		if ("replace" in $$props) $$invalidate(23, replace = $$new_props.replace);
    		if ("itemsProcessed" in $$props) $$invalidate(37, itemsProcessed = $$new_props.itemsProcessed);
    		if ("selectedLabel" in $$props) $$invalidate(24, selectedLabel = $$new_props.selectedLabel);
    		if ("filterText" in $$props) $$invalidate(38, filterText = $$new_props.filterText);
    		if ("filteredItems" in $$props) $$invalidate(25, filteredItems = $$new_props.filteredItems);
    		if ("c" in $$props) $$invalidate(26, c = $$new_props.c);
    		if ("o" in $$props) $$invalidate(27, o = $$new_props.o);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		if ($$self.$$.dirty[1] & /*items*/ 2) {
    			$$invalidate(37, itemsProcessed = process(items));
    		}

    		if ($$self.$$.dirty[0] & /*value*/ 1 | $$self.$$.dirty[1] & /*selectedLabelProp, itemsProcessed*/ 72) {
    			{
    				if (selectedLabelProp !== undefined) {
    					$$invalidate(24, selectedLabel = selectedLabelProp);
    				} else if (value !== undefined) {
    					let selectedItem = itemsProcessed.find(i => i.value === value);
    					$$invalidate(24, selectedLabel = selectedItem ? selectedItem.text : "");
    				} else {
    					$$invalidate(24, selectedLabel = "");
    				}
    			}
    		}

    		if ($$self.$$.dirty[1] & /*itemsProcessed, filterText*/ 192) {
    			$$invalidate(25, filteredItems = itemsProcessed.filter(i => !filterText || i.text.toLowerCase().includes(filterText)));
    		}

    		$$invalidate(26, c = cb.flush().add(classes, true, classesDefault).add($$props.class).get());

    		if ($$self.$$.dirty[0] & /*outlined*/ 32 | $$self.$$.dirty[1] & /*optionsClasses*/ 32) {
    			$$invalidate(27, o = ocb.flush().add(optionsClasses, true, optionsClassesDefault).add("rounded-t-none", !outlined).get());
    		}

    		if ($$self.$$.dirty[0] & /*dense*/ 1024) {
    			if (dense) {
    				$$invalidate(2, appendClasses = i => i.replace("pt-4", "pt-3"));
    			}
    		}
    	};

    	$$props = exclude_internal_props($$props);

    	return [
    		value,
    		showList,
    		appendClasses,
    		label,
    		color,
    		outlined,
    		placeholder,
    		hint,
    		error,
    		append,
    		dense,
    		persistentHint,
    		autocomplete,
    		noUnderline,
    		inputWrapperClasses,
    		labelClasses,
    		inputClasses,
    		prependClasses,
    		listClasses,
    		selectedClasses,
    		itemClasses,
    		add,
    		remove,
    		replace,
    		selectedLabel,
    		filteredItems,
    		c,
    		o,
    		dispatch,
    		filterItems,
    		handleInputClick,
    		onHideListPanel,
    		items,
    		text,
    		selectedLabelProp,
    		classes,
    		optionsClasses,
    		itemsProcessed,
    		filterText,
    		$$scope,
    		slots,
    		click_append_handler,
    		click_handler,
    		list_value_binding,
    		change_handler,
    		click_handler_1
    	];
    }

    class Select extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(
    			this,
    			options,
    			instance$5,
    			create_fragment$5,
    			safe_not_equal,
    			{
    				items: 32,
    				value: 0,
    				text: 33,
    				label: 3,
    				selectedLabel: 34,
    				color: 4,
    				outlined: 5,
    				placeholder: 6,
    				hint: 7,
    				error: 8,
    				append: 9,
    				dense: 10,
    				persistentHint: 11,
    				autocomplete: 12,
    				noUnderline: 13,
    				showList: 1,
    				classes: 35,
    				optionsClasses: 36,
    				inputWrapperClasses: 14,
    				appendClasses: 2,
    				labelClasses: 15,
    				inputClasses: 16,
    				prependClasses: 17,
    				listClasses: 18,
    				selectedClasses: 19,
    				itemClasses: 20,
    				add: 21,
    				remove: 22,
    				replace: 23
    			},
    			[-1, -1]
    		);

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Select",
    			options,
    			id: create_fragment$5.name
    		});
    	}

    	get items() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set items(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get value() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set value(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get text() {
    		return this.$$.ctx[33];
    	}

    	set text(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get label() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set label(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get selectedLabel() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set selectedLabel(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get color() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set color(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get outlined() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set outlined(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get placeholder() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set placeholder(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get hint() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set hint(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get error() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set error(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get append() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set append(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get dense() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set dense(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get persistentHint() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set persistentHint(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get autocomplete() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set autocomplete(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get noUnderline() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set noUnderline(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get showList() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set showList(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get classes() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set classes(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get optionsClasses() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set optionsClasses(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get inputWrapperClasses() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set inputWrapperClasses(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get appendClasses() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set appendClasses(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get labelClasses() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set labelClasses(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get inputClasses() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set inputClasses(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get prependClasses() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set prependClasses(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get listClasses() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set listClasses(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get selectedClasses() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set selectedClasses(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get itemClasses() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set itemClasses(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get add() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set add(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get remove() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set remove(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get replace() {
    		throw new Error("<Select>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set replace(value) {
    		throw new Error("<Select>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* node_modules\smelte\src\components\ProgressCircular\ProgressCircular.svelte generated by Svelte v3.38.2 */

    const file$3 = "node_modules\\smelte\\src\\components\\ProgressCircular\\ProgressCircular.svelte";

    function create_fragment$4(ctx) {
    	let svg;
    	let circle;
    	let circle_class_value;
    	let circle_cx_value;
    	let circle_cy_value;
    	let circle_r_value;

    	const block = {
    		c: function create() {
    			svg = svg_element("svg");
    			circle = svg_element("circle");
    			attr_dev(circle, "class", circle_class_value = "path stroke-" + /*color*/ ctx[0] + " svelte-1xkiyez");
    			attr_dev(circle, "cx", circle_cx_value = /*size*/ ctx[2] / 2);
    			attr_dev(circle, "cy", circle_cy_value = /*size*/ ctx[2] / 2);
    			attr_dev(circle, "fill", "none");
    			attr_dev(circle, "r", circle_r_value = /*size*/ ctx[2] / 2 - /*size*/ ctx[2] / 5);
    			attr_dev(circle, "stroke-width", /*width*/ ctx[1]);
    			attr_dev(circle, "stroke-miterlimit", "10");
    			attr_dev(circle, "style", /*style*/ ctx[3]);
    			add_location(circle, file$3, 49, 2, 960);
    			attr_dev(svg, "class", "circular svelte-1xkiyez");
    			set_style(svg, "width", /*size*/ ctx[2] + "px");
    			set_style(svg, "height", /*size*/ ctx[2] + "px");
    			add_location(svg, file$3, 48, 0, 892);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, svg, anchor);
    			append_dev(svg, circle);
    		},
    		p: function update(ctx, [dirty]) {
    			if (dirty & /*color*/ 1 && circle_class_value !== (circle_class_value = "path stroke-" + /*color*/ ctx[0] + " svelte-1xkiyez")) {
    				attr_dev(circle, "class", circle_class_value);
    			}

    			if (dirty & /*size*/ 4 && circle_cx_value !== (circle_cx_value = /*size*/ ctx[2] / 2)) {
    				attr_dev(circle, "cx", circle_cx_value);
    			}

    			if (dirty & /*size*/ 4 && circle_cy_value !== (circle_cy_value = /*size*/ ctx[2] / 2)) {
    				attr_dev(circle, "cy", circle_cy_value);
    			}

    			if (dirty & /*size*/ 4 && circle_r_value !== (circle_r_value = /*size*/ ctx[2] / 2 - /*size*/ ctx[2] / 5)) {
    				attr_dev(circle, "r", circle_r_value);
    			}

    			if (dirty & /*width*/ 2) {
    				attr_dev(circle, "stroke-width", /*width*/ ctx[1]);
    			}

    			if (dirty & /*style*/ 8) {
    				attr_dev(circle, "style", /*style*/ ctx[3]);
    			}

    			if (dirty & /*size*/ 4) {
    				set_style(svg, "width", /*size*/ ctx[2] + "px");
    			}

    			if (dirty & /*size*/ 4) {
    				set_style(svg, "height", /*size*/ ctx[2] + "px");
    			}
    		},
    		i: noop,
    		o: noop,
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(svg);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$4.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$4($$self, $$props, $$invalidate) {
    	let style;
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("ProgressCircular", slots, []);
    	let { progress = null } = $$props;
    	let { color = "primary" } = $$props;
    	let { width = 3 } = $$props;
    	let { size = 70 } = $$props;
    	const writable_props = ["progress", "color", "width", "size"];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== "$$") console.warn(`<ProgressCircular> was created with unknown prop '${key}'`);
    	});

    	$$self.$$set = $$props => {
    		if ("progress" in $$props) $$invalidate(4, progress = $$props.progress);
    		if ("color" in $$props) $$invalidate(0, color = $$props.color);
    		if ("width" in $$props) $$invalidate(1, width = $$props.width);
    		if ("size" in $$props) $$invalidate(2, size = $$props.size);
    	};

    	$$self.$capture_state = () => ({ progress, color, width, size, style });

    	$$self.$inject_state = $$props => {
    		if ("progress" in $$props) $$invalidate(4, progress = $$props.progress);
    		if ("color" in $$props) $$invalidate(0, color = $$props.color);
    		if ("width" in $$props) $$invalidate(1, width = $$props.width);
    		if ("size" in $$props) $$invalidate(2, size = $$props.size);
    		if ("style" in $$props) $$invalidate(3, style = $$props.style);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		if ($$self.$$.dirty & /*progress*/ 16) {
    			$$invalidate(3, style = progress > 0
    			? `
      animation: none;
      stroke-dasharray: ${150000 - progress * 1000};
      stroke-dashoffset: -${124 - progress * 124 / 100};
    `
    			: "");
    		}
    	};

    	return [color, width, size, style, progress];
    }

    class ProgressCircular extends SvelteComponentDev {
    	constructor(options) {
    		super(options);
    		init(this, options, instance$4, create_fragment$4, safe_not_equal, { progress: 4, color: 0, width: 1, size: 2 });

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "ProgressCircular",
    			options,
    			id: create_fragment$4.name
    		});
    	}

    	get progress() {
    		throw new Error("<ProgressCircular>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set progress(value) {
    		throw new Error("<ProgressCircular>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get color() {
    		throw new Error("<ProgressCircular>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set color(value) {
    		throw new Error("<ProgressCircular>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get width() {
    		throw new Error("<ProgressCircular>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set width(value) {
    		throw new Error("<ProgressCircular>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get size() {
    		throw new Error("<ProgressCircular>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set size(value) {
    		throw new Error("<ProgressCircular>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* node_modules\smelte\src\components\Slider\Slider.svelte generated by Svelte v3.38.2 */
    const file$2 = "node_modules\\smelte\\src\\components\\Slider\\Slider.svelte";

    function create_fragment$3(ctx) {
    	let label_1;
    	let t0;
    	let t1;
    	let input;
    	let mounted;
    	let dispose;

    	const block = {
    		c: function create() {
    			label_1 = element("label");
    			t0 = text(/*label*/ ctx[1]);
    			t1 = space();
    			input = element("input");
    			add_location(label_1, file$2, 53, 0, 1244);
    			attr_dev(input, "type", "range");
    			attr_dev(input, "class", /*c*/ ctx[7]);
    			attr_dev(input, "min", /*min*/ ctx[3]);
    			attr_dev(input, "max", /*max*/ ctx[4]);
    			attr_dev(input, "step", /*step*/ ctx[5]);
    			input.disabled = /*disabled*/ ctx[2];
    			attr_dev(input, "style", /*style*/ ctx[6]);
    			add_location(input, file$2, 54, 0, 1267);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, label_1, anchor);
    			append_dev(label_1, t0);
    			insert_dev(target, t1, anchor);
    			insert_dev(target, input, anchor);
    			set_input_value(input, /*value*/ ctx[0]);

    			if (!mounted) {
    				dispose = [
    					action_destroyer(/*applyColor*/ ctx[8].call(null, input)),
    					listen_dev(input, "change", /*input_change_input_handler*/ ctx[13]),
    					listen_dev(input, "input", /*input_change_input_handler*/ ctx[13]),
    					listen_dev(input, "change", /*change_handler*/ ctx[12], false, false, false)
    				];

    				mounted = true;
    			}
    		},
    		p: function update(ctx, [dirty]) {
    			if (dirty & /*label*/ 2) set_data_dev(t0, /*label*/ ctx[1]);

    			if (dirty & /*c*/ 128) {
    				attr_dev(input, "class", /*c*/ ctx[7]);
    			}

    			if (dirty & /*min*/ 8) {
    				attr_dev(input, "min", /*min*/ ctx[3]);
    			}

    			if (dirty & /*max*/ 16) {
    				attr_dev(input, "max", /*max*/ ctx[4]);
    			}

    			if (dirty & /*step*/ 32) {
    				attr_dev(input, "step", /*step*/ ctx[5]);
    			}

    			if (dirty & /*disabled*/ 4) {
    				prop_dev(input, "disabled", /*disabled*/ ctx[2]);
    			}

    			if (dirty & /*style*/ 64) {
    				attr_dev(input, "style", /*style*/ ctx[6]);
    			}

    			if (dirty & /*value*/ 1) {
    				set_input_value(input, /*value*/ ctx[0]);
    			}
    		},
    		i: noop,
    		o: noop,
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(label_1);
    			if (detaching) detach_dev(t1);
    			if (detaching) detach_dev(input);
    			mounted = false;
    			run_all(dispose);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$3.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$3($$self, $$props, $$invalidate) {
    	let c;
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("Slider", slots, []);
    	let { value = 0 } = $$props;
    	let { label = "" } = $$props;
    	let { color = "primary" } = $$props;
    	let { disabled = false } = $$props;
    	let { min = 0 } = $$props;
    	let { max = 100 } = $$props;
    	let { step = null } = $$props;
    	const classesDefault = `bg-${color}-50 w-full rounded cursor-pointer`;
    	let { classes = classesDefault } = $$props;
    	let toPercent;
    	const cb = new ClassBuilder(classes, classesDefault);
    	const getColor = c => `var(${c})`;
    	let style;

    	function applyColor(node) {
    		if (typeof window === "undefined") return false;
    		let c = getColor(`--color-${color}-500`);
    		node.style.setProperty("--bg", c);
    		node.style.setProperty("--bg-focus", c);
    	}

    	function change_handler(event) {
    		bubble($$self, event);
    	}

    	function input_change_input_handler() {
    		value = to_number(this.value);
    		$$invalidate(0, value);
    	}

    	$$self.$$set = $$new_props => {
    		$$invalidate(17, $$props = assign(assign({}, $$props), exclude_internal_props($$new_props)));
    		if ("value" in $$new_props) $$invalidate(0, value = $$new_props.value);
    		if ("label" in $$new_props) $$invalidate(1, label = $$new_props.label);
    		if ("color" in $$new_props) $$invalidate(9, color = $$new_props.color);
    		if ("disabled" in $$new_props) $$invalidate(2, disabled = $$new_props.disabled);
    		if ("min" in $$new_props) $$invalidate(3, min = $$new_props.min);
    		if ("max" in $$new_props) $$invalidate(4, max = $$new_props.max);
    		if ("step" in $$new_props) $$invalidate(5, step = $$new_props.step);
    		if ("classes" in $$new_props) $$invalidate(10, classes = $$new_props.classes);
    	};

    	$$self.$capture_state = () => ({
    		ClassBuilder,
    		Ripple,
    		value,
    		label,
    		color,
    		disabled,
    		min,
    		max,
    		step,
    		classesDefault,
    		classes,
    		toPercent,
    		cb,
    		getColor,
    		style,
    		applyColor,
    		c
    	});

    	$$self.$inject_state = $$new_props => {
    		$$invalidate(17, $$props = assign(assign({}, $$props), $$new_props));
    		if ("value" in $$props) $$invalidate(0, value = $$new_props.value);
    		if ("label" in $$props) $$invalidate(1, label = $$new_props.label);
    		if ("color" in $$props) $$invalidate(9, color = $$new_props.color);
    		if ("disabled" in $$props) $$invalidate(2, disabled = $$new_props.disabled);
    		if ("min" in $$props) $$invalidate(3, min = $$new_props.min);
    		if ("max" in $$props) $$invalidate(4, max = $$new_props.max);
    		if ("step" in $$props) $$invalidate(5, step = $$new_props.step);
    		if ("classes" in $$props) $$invalidate(10, classes = $$new_props.classes);
    		if ("toPercent" in $$props) $$invalidate(11, toPercent = $$new_props.toPercent);
    		if ("style" in $$props) $$invalidate(6, style = $$new_props.style);
    		if ("c" in $$props) $$invalidate(7, c = $$new_props.c);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		if ($$self.$$.dirty & /*max, min*/ 24) {
    			{
    				let factor = 100 / (max - min);
    				$$invalidate(11, toPercent = v => (v - min) * factor);
    			}
    		}

    		$$invalidate(7, c = cb.flush().add(classes, true, classesDefault).add($$props.class).get());

    		if ($$self.$$.dirty & /*color, toPercent, value, disabled*/ 2565) {
    			{
    				let c1 = getColor(`--color-${color}-500`);
    				let c2 = getColor(`--color-${color}-200`);
    				let cv = toPercent(value);

    				$$invalidate(6, style = disabled
    				? ""
    				: `background: linear-gradient(to right, ${c1} 0%, ${c1} ${cv}%, ${c2} ${cv}%, ${c2} 100%); --bg: ${c1}; --bg-focus: ${c1}`);
    			}
    		}
    	};

    	$$props = exclude_internal_props($$props);

    	return [
    		value,
    		label,
    		disabled,
    		min,
    		max,
    		step,
    		style,
    		c,
    		applyColor,
    		color,
    		classes,
    		toPercent,
    		change_handler,
    		input_change_input_handler
    	];
    }

    class Slider extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(this, options, instance$3, create_fragment$3, safe_not_equal, {
    			value: 0,
    			label: 1,
    			color: 9,
    			disabled: 2,
    			min: 3,
    			max: 4,
    			step: 5,
    			classes: 10
    		});

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Slider",
    			options,
    			id: create_fragment$3.name
    		});
    	}

    	get value() {
    		throw new Error("<Slider>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set value(value) {
    		throw new Error("<Slider>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get label() {
    		throw new Error("<Slider>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set label(value) {
    		throw new Error("<Slider>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get color() {
    		throw new Error("<Slider>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set color(value) {
    		throw new Error("<Slider>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get disabled() {
    		throw new Error("<Slider>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set disabled(value) {
    		throw new Error("<Slider>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get min() {
    		throw new Error("<Slider>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set min(value) {
    		throw new Error("<Slider>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get max() {
    		throw new Error("<Slider>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set max(value) {
    		throw new Error("<Slider>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get step() {
    		throw new Error("<Slider>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set step(value) {
    		throw new Error("<Slider>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get classes() {
    		throw new Error("<Slider>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set classes(value) {
    		throw new Error("<Slider>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    function withColor(color, store) {
      return message =>
        store.update(u => [
          ...u,
          {
            message,
            ts: new Date(),
            color,
            toString() {
              return message;
            }
          }
        ]);
    }

    function notificationQueue() {
      const store = writable([]);

      return {
        subscribe: store.subscribe,

        notify: withColor("gray", store),
        error: withColor("error", store),
        alert: withColor("alert", store),

        remove: i =>
          store.update(u => {
            u.splice(i, 1);
            return u;
          })
      };
    }

    /* node_modules\smelte\src\components\Snackbar\Notifications.svelte generated by Svelte v3.38.2 */

    notificationQueue();

    /* src\components\LSTMModel.svelte generated by Svelte v3.38.2 */

    const { console: console_1$1 } = globals;

    function create_fragment$2(ctx) {
    	const block = {
    		c: noop,
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: noop,
    		p: noop,
    		i: noop,
    		o: noop,
    		d: noop
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$2.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$2($$self, $$props, $$invalidate) {
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("LSTMModel", slots, []);
    	const dispatch = createEventDispatcher();

    	// Events
    	let predicting = false;

    	let training = false;
    	let error;

    	// Props
    	// Data
    	let trainingData;

    	// Model
    	let modelName = "LSTMModel";

    	let model;
    	let { inputSize = 3 } = $$props;
    	let { batchSize = 512 } = $$props; // Neuronen min 32 max 512
    	let { epochs = 10 } = $$props; // Trainings Epochen 50 iterations
    	let { optimizerName = "adam" } = $$props; // Optimizer
    	let { learningRate = 0.01 } = $$props; // Lernrate
    	let { neuronCount = 50 } = $$props;
    	let { dataLog } = $$props;

    	// lifecycle functions
    	onMount(async () => {
    		
    	});

    	// functions
    	let create = vocabularySize => {
    		const model = tf.sequential();

    		// Embedding layer
    		model.add(tf.layers.embedding({
    			inputDim: vocabularySize,
    			outputDim: inputSize,
    			inputLength: inputSize
    		}));

    		// LSTM layers
    		model.add(tf.layers.lstm({
    			units: neuronCount,
    			returnSequences: true
    		}));

    		model.add(tf.layers.lstm({
    			units: neuronCount,
    			returnSequences: false
    		}));

    		model.add(tf.layers.dropout({ rate: 0.5 }));

    		// Dense layers
    		model.add(tf.layers.dense({ units: neuronCount, activation: "relu" }));

    		model.add(tf.layers.dense({
    			units: vocabularySize,
    			activation: "softmax"
    		}));

    		console.log(model.summary());
    		return model;
    	};

    	let compile = () => {
    		const optimizer = getOptimizer(optimizerName, learningRate);

    		return model.compile({
    			optimizer,
    			loss: "sparseCategoricalCrossentropy",
    			metrics: ["accuracy"]
    		});
    	};

    	let fit = (inputs, labels) => {
    		return model.fit(inputs, labels, {
    			batchSize,
    			epochs,
    			// validationSplit: 0.3,
    			// callbacks: [dataLog],
    			// callbacks: {
    			//     onTrainBegin: (logs) => console.log("onTrainBegin:", logs),
    			//     onTrainEnd: (logs) => console.log("onTrainEnd:", logs),
    			//     onEpochBegin: (epoch, logs) => console.log("onEpochBegin:", epoch, logs),
    			//     onEpochEnd: (epoch, logs) => console.log("onEpochEnd:", epoch, logs),
    			//     onBatchBegin: (batch, logs) => console.log("onBatchBegin:", batch, logs),
    			//     onBatchEnd: (batch, logs) => console.log("onBatchEnd:", batch, logs),
    			//     onYield: (epoch, batch, logs) => console.log("onYield:", epoch, batch, logs),
    			// },
    			callbacks: tfvis.show.fitCallbacks({ name: "Training Performance" }, ["loss", "acc"], {
    				height: 200,
    				width: 400,
    				callbacks: ["onBatchEnd", "onEpochEnd"]
    			})
    		});
    	};

    	let save = async (model, name) => {
    		return await model.save(`localstorage://${name}`);
    	};

    	let load = async name => {
    		return await tf.loadLayersModel(`${name}`);
    	};

    	let prepareData = data => {
    		return tf.tidy(() => {
    			// data to lower case
    			const lowerCaseData = data.toLowerCase();

    			// data without special chars
    			const cleanData = lowerCaseData.replace(/[^a-zA-Z0-9 ]/g, "");

    			// get unique words
    			const uniqueWords = [...new Set(cleanData.split(" "))];

    			// get vocabulary size
    			const vocabularySize = uniqueWords.length + 1;

    			// clean data array
    			const cleanDataArray = cleanData.split(" ");

    			// clean data array
    			let tokenizedDataArray = [];

    			cleanDataArray.map((v, i, a) => {
    				tokenizedDataArray.push(uniqueWords.indexOf(v));
    			});

    			let trainingData = [];

    			tokenizedDataArray.map((v, i, a) => {
    				let help = [];

    				for (let j = 0; j <= inputSize; j++) {
    					help.push(a[i + j]);
    				}

    				trainingData.push(help);
    			});

    			// shuffle trainings data
    			// tf.util.shuffle(trainingData);
    			let inputs = [];

    			let labels = [];

    			// trainingData = trainingData.slice(0,85);
    			trainingData.map(e => {
    				inputs.push(e.slice(0, e.length - 1));
    				let label = e[e.length - 1];
    				labels.push(label);
    			});

    			const inputTensor = tf.tensor(inputs, [inputs.length, inputSize], "int32");
    			const labelTensor = tf.tensor(labels, [labels.length, 1], "int32");

    			return {
    				inputs: inputTensor,
    				labels: labelTensor,
    				lowerCaseData,
    				cleanData,
    				uniqueWords,
    				vocabularySize,
    				originalData: data
    			};
    		});
    	};

    	let getOptimizer = (name, learningRate) => {
    		let optimizer;

    		switch (name) {
    			case "sgd":
    				optimizer = tf.train.sgd(learningRate);
    				break;
    			case "momentum":
    				optimizer = tf.train.momentum(learningRate);
    				break;
    			case "adagrad":
    				optimizer = tf.train.adagrad(learningRate);
    				break;
    			case "adadelta":
    				optimizer = tf.train.adadelta(learningRate);
    				break;
    			case "adam":
    				optimizer = tf.train.adam(learningRate);
    				break;
    			case "adamax":
    				optimizer = tf.train.adamax(learningRate);
    				break;
    			case "rmsprop":
    				optimizer = tf.train.rmsprop(learningRate);
    				break;
    			default:
    				optimizer = tf.train.adam(learningRate);
    				break;
    		}

    		return optimizer;
    	};

    	async function train(data) {
    		dispatch("training", true);

    		// Convert the data to a form we can use for training.
    		trainingData = prepareData(data);

    		const { inputs, labels, vocabularySize } = trainingData;

    		// create model with new parms
    		model = await create(vocabularySize);

    		// Train the model
    		await compile();

    		await fit(inputs, labels);

    		// saveModel(model, modelName);
    		dispatch("training", false);
    	}

    	async function predict(inputData) {
    		dispatch("predicting", true);
    		inputData = inputData.slice(inputData.length - inputSize, inputData.length);
    		let tokenizedInputArray = inputData.map(v => trainingData.uniqueWords.indexOf(v.toLowerCase()));
    		const inputs = tf.tensor2d(tokenizedInputArray, [1, tokenizedInputArray.length]);

    		const [words] = tf.tidy(() => {
    			const pwords = model.predict(inputs);
    			return [pwords.dataSync()];
    		});

    		let results = Array.from(words).map((val, i) => {
    			return {
    				word: trainingData.uniqueWords[i],
    				acc: val
    			};
    		});

    		let sortedResults = results.sort((w1, w2) => w1.acc < w2.acc ? 1 : -1);
    		dispatch("predicting", false);
    		return sortedResults;
    	}

    	const writable_props = [
    		"inputSize",
    		"batchSize",
    		"epochs",
    		"optimizerName",
    		"learningRate",
    		"neuronCount",
    		"dataLog"
    	];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== "$$") console_1$1.warn(`<LSTMModel> was created with unknown prop '${key}'`);
    	});

    	$$self.$$set = $$props => {
    		if ("inputSize" in $$props) $$invalidate(0, inputSize = $$props.inputSize);
    		if ("batchSize" in $$props) $$invalidate(1, batchSize = $$props.batchSize);
    		if ("epochs" in $$props) $$invalidate(2, epochs = $$props.epochs);
    		if ("optimizerName" in $$props) $$invalidate(3, optimizerName = $$props.optimizerName);
    		if ("learningRate" in $$props) $$invalidate(4, learningRate = $$props.learningRate);
    		if ("neuronCount" in $$props) $$invalidate(5, neuronCount = $$props.neuronCount);
    		if ("dataLog" in $$props) $$invalidate(6, dataLog = $$props.dataLog);
    	};

    	$$self.$capture_state = () => ({
    		onMount,
    		createEventDispatcher,
    		dispatch,
    		predicting,
    		training,
    		error,
    		trainingData,
    		modelName,
    		model,
    		inputSize,
    		batchSize,
    		epochs,
    		optimizerName,
    		learningRate,
    		neuronCount,
    		dataLog,
    		create,
    		compile,
    		fit,
    		save,
    		load,
    		prepareData,
    		getOptimizer,
    		train,
    		predict
    	});

    	$$self.$inject_state = $$props => {
    		if ("predicting" in $$props) predicting = $$props.predicting;
    		if ("training" in $$props) training = $$props.training;
    		if ("error" in $$props) error = $$props.error;
    		if ("trainingData" in $$props) trainingData = $$props.trainingData;
    		if ("modelName" in $$props) modelName = $$props.modelName;
    		if ("model" in $$props) model = $$props.model;
    		if ("inputSize" in $$props) $$invalidate(0, inputSize = $$props.inputSize);
    		if ("batchSize" in $$props) $$invalidate(1, batchSize = $$props.batchSize);
    		if ("epochs" in $$props) $$invalidate(2, epochs = $$props.epochs);
    		if ("optimizerName" in $$props) $$invalidate(3, optimizerName = $$props.optimizerName);
    		if ("learningRate" in $$props) $$invalidate(4, learningRate = $$props.learningRate);
    		if ("neuronCount" in $$props) $$invalidate(5, neuronCount = $$props.neuronCount);
    		if ("dataLog" in $$props) $$invalidate(6, dataLog = $$props.dataLog);
    		if ("create" in $$props) create = $$props.create;
    		if ("compile" in $$props) compile = $$props.compile;
    		if ("fit" in $$props) fit = $$props.fit;
    		if ("save" in $$props) save = $$props.save;
    		if ("load" in $$props) load = $$props.load;
    		if ("prepareData" in $$props) prepareData = $$props.prepareData;
    		if ("getOptimizer" in $$props) getOptimizer = $$props.getOptimizer;
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	return [
    		inputSize,
    		batchSize,
    		epochs,
    		optimizerName,
    		learningRate,
    		neuronCount,
    		dataLog,
    		train,
    		predict
    	];
    }

    class LSTMModel extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(this, options, instance$2, create_fragment$2, safe_not_equal, {
    			inputSize: 0,
    			batchSize: 1,
    			epochs: 2,
    			optimizerName: 3,
    			learningRate: 4,
    			neuronCount: 5,
    			dataLog: 6,
    			train: 7,
    			predict: 8
    		});

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "LSTMModel",
    			options,
    			id: create_fragment$2.name
    		});

    		const { ctx } = this.$$;
    		const props = options.props || {};

    		if (/*dataLog*/ ctx[6] === undefined && !("dataLog" in props)) {
    			console_1$1.warn("<LSTMModel> was created without expected prop 'dataLog'");
    		}
    	}

    	get inputSize() {
    		throw new Error("<LSTMModel>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set inputSize(value) {
    		throw new Error("<LSTMModel>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get batchSize() {
    		throw new Error("<LSTMModel>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set batchSize(value) {
    		throw new Error("<LSTMModel>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get epochs() {
    		throw new Error("<LSTMModel>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set epochs(value) {
    		throw new Error("<LSTMModel>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get optimizerName() {
    		throw new Error("<LSTMModel>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set optimizerName(value) {
    		throw new Error("<LSTMModel>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get learningRate() {
    		throw new Error("<LSTMModel>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set learningRate(value) {
    		throw new Error("<LSTMModel>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get neuronCount() {
    		throw new Error("<LSTMModel>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set neuronCount(value) {
    		throw new Error("<LSTMModel>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get dataLog() {
    		throw new Error("<LSTMModel>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set dataLog(value) {
    		throw new Error("<LSTMModel>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get train() {
    		return this.$$.ctx[7];
    	}

    	set train(value) {
    		throw new Error("<LSTMModel>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get predict() {
    		return this.$$.ctx[8];
    	}

    	set predict(value) {
    		throw new Error("<LSTMModel>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* src\components\ConfigUI.svelte generated by Svelte v3.38.2 */
    const file$1 = "src\\components\\ConfigUI.svelte";

    function create_fragment$1(ctx) {
    	let div1;
    	let div0;
    	let h60;
    	let t0;
    	let t1;
    	let t2;
    	let t3;
    	let slider0;
    	let updating_value;
    	let t4;
    	let h61;
    	let t5;
    	let t6;
    	let t7;
    	let t8;
    	let slider1;
    	let updating_value_1;
    	let t9;
    	let h62;
    	let t10;
    	let t11;
    	let t12;
    	let t13;
    	let slider2;
    	let updating_value_2;
    	let t14;
    	let select;
    	let updating_value_3;
    	let t15;
    	let h63;
    	let t16;
    	let t17;
    	let t18;
    	let t19;
    	let slider3;
    	let updating_value_4;
    	let current;

    	function slider0_value_binding(value) {
    		/*slider0_value_binding*/ ctx[19](value);
    	}

    	let slider0_props = {
    		min: "32",
    		step: "10",
    		max: "512",
    		disabled: /*disabled*/ ctx[5]
    	};

    	if (/*batchSize*/ ctx[0] !== void 0) {
    		slider0_props.value = /*batchSize*/ ctx[0];
    	}

    	slider0 = new Slider({ props: slider0_props, $$inline: true });
    	binding_callbacks.push(() => bind(slider0, "value", slider0_value_binding));

    	function slider1_value_binding(value) {
    		/*slider1_value_binding*/ ctx[20](value);
    	}

    	let slider1_props = {
    		min: "10",
    		step: "10",
    		max: "1000",
    		disabled: /*disabled*/ ctx[5]
    	};

    	if (/*epochs*/ ctx[1] !== void 0) {
    		slider1_props.value = /*epochs*/ ctx[1];
    	}

    	slider1 = new Slider({ props: slider1_props, $$inline: true });
    	binding_callbacks.push(() => bind(slider1, "value", slider1_value_binding));

    	function slider2_value_binding(value) {
    		/*slider2_value_binding*/ ctx[21](value);
    	}

    	let slider2_props = {
    		min: "1",
    		step: "1",
    		max: "1000",
    		disabled: /*disabled*/ ctx[5]
    	};

    	if (/*neuronCount*/ ctx[4] !== void 0) {
    		slider2_props.value = /*neuronCount*/ ctx[4];
    	}

    	slider2 = new Slider({ props: slider2_props, $$inline: true });
    	binding_callbacks.push(() => bind(slider2, "value", slider2_value_binding));

    	function select_value_binding(value) {
    		/*select_value_binding*/ ctx[22](value);
    	}

    	let select_props = {
    		label: /*labelOptimizer*/ ctx[7],
    		items: /*optimizerList*/ ctx[11]
    	};

    	if (/*selectedOptimizer*/ ctx[2] !== void 0) {
    		select_props.value = /*selectedOptimizer*/ ctx[2];
    	}

    	select = new Select({ props: select_props, $$inline: true });
    	binding_callbacks.push(() => bind(select, "value", select_value_binding));
    	select.$on("change", /*change_handler*/ ctx[23]);

    	function slider3_value_binding(value) {
    		/*slider3_value_binding*/ ctx[24](value);
    	}

    	let slider3_props = {
    		min: ".001",
    		step: ".001",
    		max: ".1",
    		disabled: /*disabled*/ ctx[5]
    	};

    	if (/*learningRate*/ ctx[3] !== void 0) {
    		slider3_props.value = /*learningRate*/ ctx[3];
    	}

    	slider3 = new Slider({ props: slider3_props, $$inline: true });
    	binding_callbacks.push(() => bind(slider3, "value", slider3_value_binding));

    	const block = {
    		c: function create() {
    			div1 = element("div");
    			div0 = element("div");
    			h60 = element("h6");
    			t0 = text(/*labelBatchSize*/ ctx[9]);
    			t1 = text(": ");
    			t2 = text(/*batchSize*/ ctx[0]);
    			t3 = space();
    			create_component(slider0.$$.fragment);
    			t4 = space();
    			h61 = element("h6");
    			t5 = text(/*labelEpoch*/ ctx[10]);
    			t6 = text(": ");
    			t7 = text(/*epochs*/ ctx[1]);
    			t8 = space();
    			create_component(slider1.$$.fragment);
    			t9 = space();
    			h62 = element("h6");
    			t10 = text(/*labelNeurons*/ ctx[6]);
    			t11 = text(": ");
    			t12 = text(/*neuronCount*/ ctx[4]);
    			t13 = space();
    			create_component(slider2.$$.fragment);
    			t14 = space();
    			create_component(select.$$.fragment);
    			t15 = space();
    			h63 = element("h6");
    			t16 = text(/*labelLearningRate*/ ctx[8]);
    			t17 = text(": ");
    			t18 = text(/*learningRate*/ ctx[3]);
    			t19 = space();
    			create_component(slider3.$$.fragment);
    			attr_dev(h60, "class", "pt-6 pb-4");
    			add_location(h60, file$1, 66, 8, 1729);
    			attr_dev(h61, "class", "pt-6 pb-4");
    			add_location(h61, file$1, 75, 8, 1954);
    			attr_dev(h62, "class", "pt-6 pb-4");
    			add_location(h62, file$1, 78, 8, 2096);
    			attr_dev(h63, "class", "pt-6 pb-4");
    			add_location(h63, file$1, 105, 8, 2827);
    			add_location(div0, file$1, 65, 4, 1714);
    			add_location(div1, file$1, 64, 0, 1703);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div1, anchor);
    			append_dev(div1, div0);
    			append_dev(div0, h60);
    			append_dev(h60, t0);
    			append_dev(h60, t1);
    			append_dev(h60, t2);
    			append_dev(div0, t3);
    			mount_component(slider0, div0, null);
    			append_dev(div0, t4);
    			append_dev(div0, h61);
    			append_dev(h61, t5);
    			append_dev(h61, t6);
    			append_dev(h61, t7);
    			append_dev(div0, t8);
    			mount_component(slider1, div0, null);
    			append_dev(div0, t9);
    			append_dev(div0, h62);
    			append_dev(h62, t10);
    			append_dev(h62, t11);
    			append_dev(h62, t12);
    			append_dev(div0, t13);
    			mount_component(slider2, div0, null);
    			append_dev(div0, t14);
    			mount_component(select, div0, null);
    			append_dev(div0, t15);
    			append_dev(div0, h63);
    			append_dev(h63, t16);
    			append_dev(h63, t17);
    			append_dev(h63, t18);
    			append_dev(div0, t19);
    			mount_component(slider3, div0, null);
    			current = true;
    		},
    		p: function update(ctx, [dirty]) {
    			if (!current || dirty & /*batchSize*/ 1) set_data_dev(t2, /*batchSize*/ ctx[0]);
    			const slider0_changes = {};
    			if (dirty & /*disabled*/ 32) slider0_changes.disabled = /*disabled*/ ctx[5];

    			if (!updating_value && dirty & /*batchSize*/ 1) {
    				updating_value = true;
    				slider0_changes.value = /*batchSize*/ ctx[0];
    				add_flush_callback(() => updating_value = false);
    			}

    			slider0.$set(slider0_changes);
    			if (!current || dirty & /*epochs*/ 2) set_data_dev(t7, /*epochs*/ ctx[1]);
    			const slider1_changes = {};
    			if (dirty & /*disabled*/ 32) slider1_changes.disabled = /*disabled*/ ctx[5];

    			if (!updating_value_1 && dirty & /*epochs*/ 2) {
    				updating_value_1 = true;
    				slider1_changes.value = /*epochs*/ ctx[1];
    				add_flush_callback(() => updating_value_1 = false);
    			}

    			slider1.$set(slider1_changes);
    			if (!current || dirty & /*neuronCount*/ 16) set_data_dev(t12, /*neuronCount*/ ctx[4]);
    			const slider2_changes = {};
    			if (dirty & /*disabled*/ 32) slider2_changes.disabled = /*disabled*/ ctx[5];

    			if (!updating_value_2 && dirty & /*neuronCount*/ 16) {
    				updating_value_2 = true;
    				slider2_changes.value = /*neuronCount*/ ctx[4];
    				add_flush_callback(() => updating_value_2 = false);
    			}

    			slider2.$set(slider2_changes);
    			const select_changes = {};

    			if (!updating_value_3 && dirty & /*selectedOptimizer*/ 4) {
    				updating_value_3 = true;
    				select_changes.value = /*selectedOptimizer*/ ctx[2];
    				add_flush_callback(() => updating_value_3 = false);
    			}

    			select.$set(select_changes);
    			if (!current || dirty & /*learningRate*/ 8) set_data_dev(t18, /*learningRate*/ ctx[3]);
    			const slider3_changes = {};
    			if (dirty & /*disabled*/ 32) slider3_changes.disabled = /*disabled*/ ctx[5];

    			if (!updating_value_4 && dirty & /*learningRate*/ 8) {
    				updating_value_4 = true;
    				slider3_changes.value = /*learningRate*/ ctx[3];
    				add_flush_callback(() => updating_value_4 = false);
    			}

    			slider3.$set(slider3_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(slider0.$$.fragment, local);
    			transition_in(slider1.$$.fragment, local);
    			transition_in(slider2.$$.fragment, local);
    			transition_in(select.$$.fragment, local);
    			transition_in(slider3.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(slider0.$$.fragment, local);
    			transition_out(slider1.$$.fragment, local);
    			transition_out(slider2.$$.fragment, local);
    			transition_out(select.$$.fragment, local);
    			transition_out(slider3.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div1);
    			destroy_component(slider0);
    			destroy_component(slider1);
    			destroy_component(slider2);
    			destroy_component(select);
    			destroy_component(slider3);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$1.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$1($$self, $$props, $$invalidate) {
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("ConfigUI", slots, []);
    	let labelSettings = "Settings";
    	let labelNeurons = "Neurons";
    	let labelActivationFunction = "activation";
    	let labelOptimizer = "optimizer";
    	let labelLearningRate = "learning rate";
    	let labelBatchSize = "batch size";
    	let labelEpoch = "epoch";
    	let labelHiddenLayer = "hidden layer";
    	let labelMinWeight = "min weight";
    	let labelMaxWeight = "max weight";
    	let { name = "Model 1" } = $$props;
    	let { disabled = false } = $$props;
    	let { batchSize = 100 } = $$props; // Neuronen min 32 max 512
    	let { epochs = 200 } = $$props; // Trainings Epochen 50 iterations
    	let { hiddenLayerCount = 10 } = $$props; // Anzahl der hidden Layer
    	let { stepWeight = 0.01 } = $$props;
    	let { minWeight = 0 } = $$props;
    	let { maxWeight = 1 } = $$props;
    	let { maxMaxWeight = 1 } = $$props;
    	let { activationFunction = "none" } = $$props;
    	let { selectedOptimizer = "sgd" } = $$props; // Optimizer
    	let { learningRate = 0.01 } = $$props; // Lernrate
    	let { neuronCount = 100 } = $$props;

    	const activationList = [
    		"none",
    		"elu",
    		"hardSigmoid",
    		"linear",
    		"relu",
    		"relu6",
    		"selu",
    		"sigmoid",
    		"softmax",
    		"softplus",
    		"softsign",
    		"tanh",
    		"swish",
    		"mish"
    	];

    	const optimizerList = ["sgd", "momentum", "adagrad", "adadelta", "adam", "adamax", "rmsprop"];

    	// lifecycle functions
    	onMount(async () => {
    		
    	});

    	const writable_props = [
    		"name",
    		"disabled",
    		"batchSize",
    		"epochs",
    		"hiddenLayerCount",
    		"stepWeight",
    		"minWeight",
    		"maxWeight",
    		"maxMaxWeight",
    		"activationFunction",
    		"selectedOptimizer",
    		"learningRate",
    		"neuronCount"
    	];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== "$$") console.warn(`<ConfigUI> was created with unknown prop '${key}'`);
    	});

    	function slider0_value_binding(value) {
    		batchSize = value;
    		$$invalidate(0, batchSize);
    	}

    	function slider1_value_binding(value) {
    		epochs = value;
    		$$invalidate(1, epochs);
    	}

    	function slider2_value_binding(value) {
    		neuronCount = value;
    		$$invalidate(4, neuronCount);
    	}

    	function select_value_binding(value) {
    		selectedOptimizer = value;
    		$$invalidate(2, selectedOptimizer);
    	}

    	const change_handler = v => {
    		$$invalidate(2, selectedOptimizer = v.detail);
    	};

    	function slider3_value_binding(value) {
    		learningRate = value;
    		$$invalidate(3, learningRate);
    	}

    	$$self.$$set = $$props => {
    		if ("name" in $$props) $$invalidate(12, name = $$props.name);
    		if ("disabled" in $$props) $$invalidate(5, disabled = $$props.disabled);
    		if ("batchSize" in $$props) $$invalidate(0, batchSize = $$props.batchSize);
    		if ("epochs" in $$props) $$invalidate(1, epochs = $$props.epochs);
    		if ("hiddenLayerCount" in $$props) $$invalidate(13, hiddenLayerCount = $$props.hiddenLayerCount);
    		if ("stepWeight" in $$props) $$invalidate(14, stepWeight = $$props.stepWeight);
    		if ("minWeight" in $$props) $$invalidate(15, minWeight = $$props.minWeight);
    		if ("maxWeight" in $$props) $$invalidate(16, maxWeight = $$props.maxWeight);
    		if ("maxMaxWeight" in $$props) $$invalidate(17, maxMaxWeight = $$props.maxMaxWeight);
    		if ("activationFunction" in $$props) $$invalidate(18, activationFunction = $$props.activationFunction);
    		if ("selectedOptimizer" in $$props) $$invalidate(2, selectedOptimizer = $$props.selectedOptimizer);
    		if ("learningRate" in $$props) $$invalidate(3, learningRate = $$props.learningRate);
    		if ("neuronCount" in $$props) $$invalidate(4, neuronCount = $$props.neuronCount);
    	};

    	$$self.$capture_state = () => ({
    		onMount,
    		Slider,
    		Button,
    		ProgressCircular,
    		Select,
    		labelSettings,
    		labelNeurons,
    		labelActivationFunction,
    		labelOptimizer,
    		labelLearningRate,
    		labelBatchSize,
    		labelEpoch,
    		labelHiddenLayer,
    		labelMinWeight,
    		labelMaxWeight,
    		name,
    		disabled,
    		batchSize,
    		epochs,
    		hiddenLayerCount,
    		stepWeight,
    		minWeight,
    		maxWeight,
    		maxMaxWeight,
    		activationFunction,
    		selectedOptimizer,
    		learningRate,
    		neuronCount,
    		activationList,
    		optimizerList
    	});

    	$$self.$inject_state = $$props => {
    		if ("labelSettings" in $$props) labelSettings = $$props.labelSettings;
    		if ("labelNeurons" in $$props) $$invalidate(6, labelNeurons = $$props.labelNeurons);
    		if ("labelActivationFunction" in $$props) labelActivationFunction = $$props.labelActivationFunction;
    		if ("labelOptimizer" in $$props) $$invalidate(7, labelOptimizer = $$props.labelOptimizer);
    		if ("labelLearningRate" in $$props) $$invalidate(8, labelLearningRate = $$props.labelLearningRate);
    		if ("labelBatchSize" in $$props) $$invalidate(9, labelBatchSize = $$props.labelBatchSize);
    		if ("labelEpoch" in $$props) $$invalidate(10, labelEpoch = $$props.labelEpoch);
    		if ("labelHiddenLayer" in $$props) labelHiddenLayer = $$props.labelHiddenLayer;
    		if ("labelMinWeight" in $$props) labelMinWeight = $$props.labelMinWeight;
    		if ("labelMaxWeight" in $$props) labelMaxWeight = $$props.labelMaxWeight;
    		if ("name" in $$props) $$invalidate(12, name = $$props.name);
    		if ("disabled" in $$props) $$invalidate(5, disabled = $$props.disabled);
    		if ("batchSize" in $$props) $$invalidate(0, batchSize = $$props.batchSize);
    		if ("epochs" in $$props) $$invalidate(1, epochs = $$props.epochs);
    		if ("hiddenLayerCount" in $$props) $$invalidate(13, hiddenLayerCount = $$props.hiddenLayerCount);
    		if ("stepWeight" in $$props) $$invalidate(14, stepWeight = $$props.stepWeight);
    		if ("minWeight" in $$props) $$invalidate(15, minWeight = $$props.minWeight);
    		if ("maxWeight" in $$props) $$invalidate(16, maxWeight = $$props.maxWeight);
    		if ("maxMaxWeight" in $$props) $$invalidate(17, maxMaxWeight = $$props.maxMaxWeight);
    		if ("activationFunction" in $$props) $$invalidate(18, activationFunction = $$props.activationFunction);
    		if ("selectedOptimizer" in $$props) $$invalidate(2, selectedOptimizer = $$props.selectedOptimizer);
    		if ("learningRate" in $$props) $$invalidate(3, learningRate = $$props.learningRate);
    		if ("neuronCount" in $$props) $$invalidate(4, neuronCount = $$props.neuronCount);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	return [
    		batchSize,
    		epochs,
    		selectedOptimizer,
    		learningRate,
    		neuronCount,
    		disabled,
    		labelNeurons,
    		labelOptimizer,
    		labelLearningRate,
    		labelBatchSize,
    		labelEpoch,
    		optimizerList,
    		name,
    		hiddenLayerCount,
    		stepWeight,
    		minWeight,
    		maxWeight,
    		maxMaxWeight,
    		activationFunction,
    		slider0_value_binding,
    		slider1_value_binding,
    		slider2_value_binding,
    		select_value_binding,
    		change_handler,
    		slider3_value_binding
    	];
    }

    class ConfigUI extends SvelteComponentDev {
    	constructor(options) {
    		super(options);

    		init(this, options, instance$1, create_fragment$1, safe_not_equal, {
    			name: 12,
    			disabled: 5,
    			batchSize: 0,
    			epochs: 1,
    			hiddenLayerCount: 13,
    			stepWeight: 14,
    			minWeight: 15,
    			maxWeight: 16,
    			maxMaxWeight: 17,
    			activationFunction: 18,
    			selectedOptimizer: 2,
    			learningRate: 3,
    			neuronCount: 4
    		});

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "ConfigUI",
    			options,
    			id: create_fragment$1.name
    		});
    	}

    	get name() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set name(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get disabled() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set disabled(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get batchSize() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set batchSize(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get epochs() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set epochs(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get hiddenLayerCount() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set hiddenLayerCount(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get stepWeight() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set stepWeight(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get minWeight() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set minWeight(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get maxWeight() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set maxWeight(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get maxMaxWeight() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set maxMaxWeight(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get activationFunction() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set activationFunction(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get selectedOptimizer() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set selectedOptimizer(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get learningRate() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set learningRate(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get neuronCount() {
    		throw new Error("<ConfigUI>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set neuronCount(value) {
    		throw new Error("<ConfigUI>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* src\App.svelte generated by Svelte v3.38.2 */

    const { console: console_1 } = globals;
    const file = "src\\App.svelte";

    // (237:3) <Button      block      outlined      on:click={train}      disabled={!trainingData || modelIsWorking}>
    function create_default_slot(ctx) {
    	let t;

    	const block = {
    		c: function create() {
    			t = text("train");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, t, anchor);
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(t);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot.name,
    		type: "slot",
    		source: "(237:3) <Button      block      outlined      on:click={train}      disabled={!trainingData || modelIsWorking}>",
    		ctx
    	});

    	return block;
    }

    // (247:2) {:else}
    function create_else_block(ctx) {
    	let div;
    	let h5;
    	let t1;
    	let textfield;
    	let updating_value;
    	let t2;
    	let t3;
    	let current;

    	function textfield_value_binding(value) {
    		/*textfield_value_binding*/ ctx[48](value);
    	}

    	let textfield_props = {
    		label: /*labelPredictionInput*/ ctx[27],
    		disabled: !/*trained*/ ctx[7],
    		outlined: true
    	};

    	if (/*input*/ ctx[4] !== void 0) {
    		textfield_props.value = /*input*/ ctx[4];
    	}

    	textfield = new TextField({ props: textfield_props, $$inline: true });
    	binding_callbacks.push(() => bind(textfield, "value", textfield_value_binding));
    	textfield.$on("input", /*predictInput*/ ctx[31]);
    	let if_block0 = /*predicting*/ ctx[3] && create_if_block_2(ctx);
    	let if_block1 = !/*predicting*/ ctx[3] && /*predictedItems*/ ctx[5].length > 0 && create_if_block_1(ctx);

    	const block = {
    		c: function create() {
    			div = element("div");
    			h5 = element("h5");
    			h5.textContent = `${/*labelPredictionInputTitle*/ ctx[26]}`;
    			t1 = space();
    			create_component(textfield.$$.fragment);
    			t2 = space();
    			if (if_block0) if_block0.c();
    			t3 = space();
    			if (if_block1) if_block1.c();
    			attr_dev(h5, "class", "pt-6 pb-4 svelte-1k8gf66");
    			add_location(h5, file, 248, 4, 6120);
    			attr_dev(div, "class", "svelte-1k8gf66");
    			add_location(div, file, 247, 3, 6109);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div, anchor);
    			append_dev(div, h5);
    			append_dev(div, t1);
    			mount_component(textfield, div, null);
    			append_dev(div, t2);
    			if (if_block0) if_block0.m(div, null);
    			append_dev(div, t3);
    			if (if_block1) if_block1.m(div, null);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const textfield_changes = {};
    			if (dirty[0] & /*trained*/ 128) textfield_changes.disabled = !/*trained*/ ctx[7];

    			if (!updating_value && dirty[0] & /*input*/ 16) {
    				updating_value = true;
    				textfield_changes.value = /*input*/ ctx[4];
    				add_flush_callback(() => updating_value = false);
    			}

    			textfield.$set(textfield_changes);

    			if (/*predicting*/ ctx[3]) {
    				if (if_block0) {
    					if (dirty[0] & /*predicting*/ 8) {
    						transition_in(if_block0, 1);
    					}
    				} else {
    					if_block0 = create_if_block_2(ctx);
    					if_block0.c();
    					transition_in(if_block0, 1);
    					if_block0.m(div, t3);
    				}
    			} else if (if_block0) {
    				group_outros();

    				transition_out(if_block0, 1, 1, () => {
    					if_block0 = null;
    				});

    				check_outros();
    			}

    			if (!/*predicting*/ ctx[3] && /*predictedItems*/ ctx[5].length > 0) {
    				if (if_block1) {
    					if_block1.p(ctx, dirty);

    					if (dirty[0] & /*predicting, predictedItems*/ 40) {
    						transition_in(if_block1, 1);
    					}
    				} else {
    					if_block1 = create_if_block_1(ctx);
    					if_block1.c();
    					transition_in(if_block1, 1);
    					if_block1.m(div, null);
    				}
    			} else if (if_block1) {
    				group_outros();

    				transition_out(if_block1, 1, 1, () => {
    					if_block1 = null;
    				});

    				check_outros();
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(textfield.$$.fragment, local);
    			transition_in(if_block0);
    			transition_in(if_block1);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(textfield.$$.fragment, local);
    			transition_out(if_block0);
    			transition_out(if_block1);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div);
    			destroy_component(textfield);
    			if (if_block0) if_block0.d();
    			if (if_block1) if_block1.d();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_else_block.name,
    		type: "else",
    		source: "(247:2) {:else}",
    		ctx
    	});

    	return block;
    }

    // (244:2) {#if modelIsWorking}
    function create_if_block(ctx) {
    	let progresscircular;
    	let current;
    	progresscircular = new ProgressCircular({ $$inline: true });

    	const block = {
    		c: function create() {
    			create_component(progresscircular.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(progresscircular, target, anchor);
    			current = true;
    		},
    		p: noop,
    		i: function intro(local) {
    			if (current) return;
    			transition_in(progresscircular.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(progresscircular.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(progresscircular, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block.name,
    		type: "if",
    		source: "(244:2) {#if modelIsWorking}",
    		ctx
    	});

    	return block;
    }

    // (257:4) {#if predicting}
    function create_if_block_2(ctx) {
    	let progresscircular;
    	let current;
    	progresscircular = new ProgressCircular({ $$inline: true });

    	const block = {
    		c: function create() {
    			create_component(progresscircular.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(progresscircular, target, anchor);
    			current = true;
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(progresscircular.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(progresscircular.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(progresscircular, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_2.name,
    		type: "if",
    		source: "(257:4) {#if predicting}",
    		ctx
    	});

    	return block;
    }

    // (260:4) {#if !predicting && predictedItems.length > 0}
    function create_if_block_1(ctx) {
    	let h7;
    	let t1;
    	let list;
    	let current;

    	list = new List({
    			props: { items: /*predictedItems*/ ctx[5] },
    			$$inline: true
    		});

    	list.$on("change", /*selectItem*/ ctx[32]);

    	const block = {
    		c: function create() {
    			h7 = element("h7");
    			h7.textContent = `${/*labelPredictedItemsTitle*/ ctx[28]}`;
    			t1 = space();
    			create_component(list.$$.fragment);
    			attr_dev(h7, "class", "svelte-1k8gf66");
    			add_location(h7, file, 260, 5, 6448);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, h7, anchor);
    			insert_dev(target, t1, anchor);
    			mount_component(list, target, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const list_changes = {};
    			if (dirty[0] & /*predictedItems*/ 32) list_changes.items = /*predictedItems*/ ctx[5];
    			list.$set(list_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(list.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(list.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(h7);
    			if (detaching) detach_dev(t1);
    			destroy_component(list, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_if_block_1.name,
    		type: "if",
    		source: "(260:4) {#if !predicting && predictedItems.length > 0}",
    		ctx
    	});

    	return block;
    }

    function create_fragment(ctx) {
    	let header;
    	let h50;
    	let t2;
    	let h3;
    	let t4;
    	let main;
    	let lstmmodel;
    	let t5;
    	let div3;
    	let div0;
    	let h51;
    	let t7;
    	let configui;
    	let updating_name;
    	let updating_batchSize;
    	let updating_epochs;
    	let updating_hiddenLayerCount;
    	let updating_activationFunction;
    	let updating_selectedOptimizer;
    	let updating_learningRate;
    	let updating_neuronCount;
    	let t8;
    	let textfield0;
    	let updating_value;
    	let t9;
    	let textfield1;
    	let updating_value_1;
    	let t10;
    	let div1;
    	let h52;
    	let t12;
    	let select;
    	let t13;
    	let textfield2;
    	let updating_value_2;
    	let t14;
    	let button;
    	let t15;
    	let current_block_type_index;
    	let if_block;
    	let t16;
    	let div2;
    	let zero_md;
    	let zero_md_src_value;
    	let t17;
    	let footer;
    	let div4;
    	let h53;
    	let t19;
    	let a0;
    	let p0;
    	let t21;
    	let a1;
    	let p1;
    	let t23;
    	let a2;
    	let p2;
    	let t25;
    	let a3;
    	let p3;
    	let current;

    	let lstmmodel_props = {
    		modelName: /*modelName*/ ctx[1],
    		batchSize: /*batchSize*/ ctx[8],
    		inputSize: /*inputSize*/ ctx[15],
    		epochs: /*epochs*/ ctx[9],
    		selectedOptimizer: /*selectedOptimizer*/ ctx[12],
    		learningRate: /*learningRate*/ ctx[13],
    		neuronCount: /*neuronCount*/ ctx[14]
    	};

    	lstmmodel = new LSTMModel({ props: lstmmodel_props, $$inline: true });
    	/*lstmmodel_binding*/ ctx[33](lstmmodel);
    	lstmmodel.$on("predicting", /*predicting_handler*/ ctx[34]);
    	lstmmodel.$on("training", /*training_handler*/ ctx[35]);

    	function configui_name_binding(value) {
    		/*configui_name_binding*/ ctx[36](value);
    	}

    	function configui_batchSize_binding(value) {
    		/*configui_batchSize_binding*/ ctx[37](value);
    	}

    	function configui_epochs_binding(value) {
    		/*configui_epochs_binding*/ ctx[38](value);
    	}

    	function configui_hiddenLayerCount_binding(value) {
    		/*configui_hiddenLayerCount_binding*/ ctx[39](value);
    	}

    	function configui_activationFunction_binding(value) {
    		/*configui_activationFunction_binding*/ ctx[40](value);
    	}

    	function configui_selectedOptimizer_binding(value) {
    		/*configui_selectedOptimizer_binding*/ ctx[41](value);
    	}

    	function configui_learningRate_binding(value) {
    		/*configui_learningRate_binding*/ ctx[42](value);
    	}

    	function configui_neuronCount_binding(value) {
    		/*configui_neuronCount_binding*/ ctx[43](value);
    	}

    	let configui_props = { disabled: /*modelIsWorking*/ ctx[2] };

    	if (/*modelName*/ ctx[1] !== void 0) {
    		configui_props.name = /*modelName*/ ctx[1];
    	}

    	if (/*batchSize*/ ctx[8] !== void 0) {
    		configui_props.batchSize = /*batchSize*/ ctx[8];
    	}

    	if (/*epochs*/ ctx[9] !== void 0) {
    		configui_props.epochs = /*epochs*/ ctx[9];
    	}

    	if (/*hiddenLayerCount*/ ctx[10] !== void 0) {
    		configui_props.hiddenLayerCount = /*hiddenLayerCount*/ ctx[10];
    	}

    	if (/*activationFunction*/ ctx[11] !== void 0) {
    		configui_props.activationFunction = /*activationFunction*/ ctx[11];
    	}

    	if (/*selectedOptimizer*/ ctx[12] !== void 0) {
    		configui_props.selectedOptimizer = /*selectedOptimizer*/ ctx[12];
    	}

    	if (/*learningRate*/ ctx[13] !== void 0) {
    		configui_props.learningRate = /*learningRate*/ ctx[13];
    	}

    	if (/*neuronCount*/ ctx[14] !== void 0) {
    		configui_props.neuronCount = /*neuronCount*/ ctx[14];
    	}

    	configui = new ConfigUI({ props: configui_props, $$inline: true });
    	binding_callbacks.push(() => bind(configui, "name", configui_name_binding));
    	binding_callbacks.push(() => bind(configui, "batchSize", configui_batchSize_binding));
    	binding_callbacks.push(() => bind(configui, "epochs", configui_epochs_binding));
    	binding_callbacks.push(() => bind(configui, "hiddenLayerCount", configui_hiddenLayerCount_binding));
    	binding_callbacks.push(() => bind(configui, "activationFunction", configui_activationFunction_binding));
    	binding_callbacks.push(() => bind(configui, "selectedOptimizer", configui_selectedOptimizer_binding));
    	binding_callbacks.push(() => bind(configui, "learningRate", configui_learningRate_binding));
    	binding_callbacks.push(() => bind(configui, "neuronCount", configui_neuronCount_binding));

    	function textfield0_value_binding(value) {
    		/*textfield0_value_binding*/ ctx[44](value);
    	}

    	let textfield0_props = {
    		label: /*labelWordsForPrediction*/ ctx[22],
    		outlined: true
    	};

    	if (/*inputSize*/ ctx[15] !== void 0) {
    		textfield0_props.value = /*inputSize*/ ctx[15];
    	}

    	textfield0 = new TextField({ props: textfield0_props, $$inline: true });
    	binding_callbacks.push(() => bind(textfield0, "value", textfield0_value_binding));

    	function textfield1_value_binding(value) {
    		/*textfield1_value_binding*/ ctx[45](value);
    	}

    	let textfield1_props = {
    		label: /*labelShowResults*/ ctx[23],
    		outlined: true
    	};

    	if (/*maxResultSize*/ ctx[6] !== void 0) {
    		textfield1_props.value = /*maxResultSize*/ ctx[6];
    	}

    	textfield1 = new TextField({ props: textfield1_props, $$inline: true });
    	binding_callbacks.push(() => bind(textfield1, "value", textfield1_value_binding));

    	select = new Select({
    			props: {
    				label: /*labelDatasetsTitle*/ ctx[24],
    				items: /*trainingDataSets*/ ctx[16],
    				disabled: /*modelIsWorking*/ ctx[2]
    			},
    			$$inline: true
    		});

    	select.$on("change", /*change_handler*/ ctx[46]);

    	function textfield2_value_binding(value) {
    		/*textfield2_value_binding*/ ctx[47](value);
    	}

    	let textfield2_props = {
    		label: /*labelDatasetsText*/ ctx[25],
    		textarea: true,
    		rows: "5",
    		outlined: true,
    		disabled: true
    	};

    	if (/*previewText*/ ctx[18] !== void 0) {
    		textfield2_props.value = /*previewText*/ ctx[18];
    	}

    	textfield2 = new TextField({ props: textfield2_props, $$inline: true });
    	binding_callbacks.push(() => bind(textfield2, "value", textfield2_value_binding));

    	button = new Button({
    			props: {
    				block: true,
    				outlined: true,
    				disabled: !/*trainingData*/ ctx[17] || /*modelIsWorking*/ ctx[2],
    				$$slots: { default: [create_default_slot] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	button.$on("click", /*train*/ ctx[30]);
    	const if_block_creators = [create_if_block, create_else_block];
    	const if_blocks = [];

    	function select_block_type(ctx, dirty) {
    		if (/*modelIsWorking*/ ctx[2]) return 0;
    		return 1;
    	}

    	current_block_type_index = select_block_type(ctx);
    	if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);

    	const block = {
    		c: function create() {
    			header = element("header");
    			h50 = element("h5");
    			h50.textContent = `Einsendeaufgabe ${/*taskNumber*/ ctx[20]}`;
    			t2 = space();
    			h3 = element("h3");
    			h3.textContent = `${/*taskTitle*/ ctx[19]}`;
    			t4 = space();
    			main = element("main");
    			create_component(lstmmodel.$$.fragment);
    			t5 = space();
    			div3 = element("div");
    			div0 = element("div");
    			h51 = element("h5");
    			h51.textContent = `${/*labelSettings*/ ctx[21]}`;
    			t7 = space();
    			create_component(configui.$$.fragment);
    			t8 = space();
    			create_component(textfield0.$$.fragment);
    			t9 = space();
    			create_component(textfield1.$$.fragment);
    			t10 = space();
    			div1 = element("div");
    			h52 = element("h5");
    			h52.textContent = `${/*labelDatasetsTitle*/ ctx[24]}`;
    			t12 = space();
    			create_component(select.$$.fragment);
    			t13 = space();
    			create_component(textfield2.$$.fragment);
    			t14 = space();
    			create_component(button.$$.fragment);
    			t15 = space();
    			if_block.c();
    			t16 = space();
    			div2 = element("div");
    			zero_md = element("zero-md");
    			t17 = space();
    			footer = element("footer");
    			div4 = element("div");
    			h53 = element("h5");
    			h53.textContent = "Ressourcen";
    			t19 = space();
    			a0 = element("a");
    			p0 = element("p");
    			p0.textContent = "Github Repository";
    			t21 = space();
    			a1 = element("a");
    			p1 = element("p");
    			p1.textContent = "Svelte";
    			t23 = space();
    			a2 = element("a");
    			p2 = element("p");
    			p2.textContent = "Smeltejs";
    			t25 = space();
    			a3 = element("a");
    			p3 = element("p");
    			p3.textContent = "Pleanarprotokoll Deutscher Bundestag";
    			attr_dev(h50, "class", "svelte-1k8gf66");
    			add_location(h50, file, 167, 1, 4360);
    			attr_dev(h3, "class", "svelte-1k8gf66");
    			add_location(h3, file, 168, 1, 4400);
    			attr_dev(header, "class", "svelte-1k8gf66");
    			add_location(header, file, 166, 0, 4349);
    			attr_dev(h51, "class", "pb-4 svelte-1k8gf66");
    			add_location(h51, file, 192, 3, 4892);
    			attr_dev(div0, "class", "svelte-1k8gf66");
    			add_location(div0, file, 191, 2, 4882);
    			attr_dev(h52, "class", "pb-4 svelte-1k8gf66");
    			add_location(h52, file, 216, 3, 5397);
    			attr_dev(div1, "class", "svelte-1k8gf66");
    			add_location(div1, file, 215, 2, 5387);
    			if (zero_md.src !== (zero_md_src_value = /*mdUrl*/ ctx[29])) set_custom_element_data(zero_md, "src", zero_md_src_value);
    			set_custom_element_data(zero_md, "class", "svelte-1k8gf66");
    			add_location(zero_md, file, 266, 3, 6589);
    			attr_dev(div2, "class", "svelte-1k8gf66");
    			add_location(div2, file, 265, 2, 6579);
    			attr_dev(div3, "class", "grid svelte-1k8gf66");
    			add_location(div3, file, 190, 1, 4860);
    			attr_dev(main, "class", "svelte-1k8gf66");
    			add_location(main, file, 171, 0, 4435);
    			attr_dev(h53, "class", "svelte-1k8gf66");
    			add_location(h53, file, 273, 2, 6664);
    			attr_dev(p0, "class", "svelte-1k8gf66");
    			add_location(p0, file, 275, 3, 6740);
    			attr_dev(a0, "href", "https://github.com/tonyflow90/dl_ea03");
    			attr_dev(a0, "class", "svelte-1k8gf66");
    			add_location(a0, file, 274, 2, 6687);
    			attr_dev(p1, "class", "svelte-1k8gf66");
    			add_location(p1, file, 278, 3, 6811);
    			attr_dev(a1, "href", "https://svelte.dev/");
    			attr_dev(a1, "class", "svelte-1k8gf66");
    			add_location(a1, file, 277, 2, 6776);
    			attr_dev(p2, "class", "svelte-1k8gf66");
    			add_location(p2, file, 281, 3, 6873);
    			attr_dev(a2, "href", "https://smeltejs.com/");
    			attr_dev(a2, "class", "svelte-1k8gf66");
    			add_location(a2, file, 280, 2, 6836);
    			attr_dev(p3, "class", "svelte-1k8gf66");
    			add_location(p3, file, 284, 3, 6958);
    			attr_dev(a3, "href", "https://www.bundestag.de/services/opendata");
    			attr_dev(a3, "class", "svelte-1k8gf66");
    			add_location(a3, file, 283, 2, 6900);
    			attr_dev(div4, "class", "svelte-1k8gf66");
    			add_location(div4, file, 272, 1, 6655);
    			attr_dev(footer, "class", "svelte-1k8gf66");
    			add_location(footer, file, 271, 0, 6644);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, header, anchor);
    			append_dev(header, h50);
    			append_dev(header, t2);
    			append_dev(header, h3);
    			insert_dev(target, t4, anchor);
    			insert_dev(target, main, anchor);
    			mount_component(lstmmodel, main, null);
    			append_dev(main, t5);
    			append_dev(main, div3);
    			append_dev(div3, div0);
    			append_dev(div0, h51);
    			append_dev(div0, t7);
    			mount_component(configui, div0, null);
    			append_dev(div0, t8);
    			mount_component(textfield0, div0, null);
    			append_dev(div0, t9);
    			mount_component(textfield1, div0, null);
    			append_dev(div3, t10);
    			append_dev(div3, div1);
    			append_dev(div1, h52);
    			append_dev(div1, t12);
    			mount_component(select, div1, null);
    			append_dev(div1, t13);
    			mount_component(textfield2, div1, null);
    			append_dev(div1, t14);
    			mount_component(button, div1, null);
    			append_dev(div3, t15);
    			if_blocks[current_block_type_index].m(div3, null);
    			append_dev(div3, t16);
    			append_dev(div3, div2);
    			append_dev(div2, zero_md);
    			insert_dev(target, t17, anchor);
    			insert_dev(target, footer, anchor);
    			append_dev(footer, div4);
    			append_dev(div4, h53);
    			append_dev(div4, t19);
    			append_dev(div4, a0);
    			append_dev(a0, p0);
    			append_dev(div4, t21);
    			append_dev(div4, a1);
    			append_dev(a1, p1);
    			append_dev(div4, t23);
    			append_dev(div4, a2);
    			append_dev(a2, p2);
    			append_dev(div4, t25);
    			append_dev(div4, a3);
    			append_dev(a3, p3);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const lstmmodel_changes = {};
    			if (dirty[0] & /*modelName*/ 2) lstmmodel_changes.modelName = /*modelName*/ ctx[1];
    			if (dirty[0] & /*batchSize*/ 256) lstmmodel_changes.batchSize = /*batchSize*/ ctx[8];
    			if (dirty[0] & /*inputSize*/ 32768) lstmmodel_changes.inputSize = /*inputSize*/ ctx[15];
    			if (dirty[0] & /*epochs*/ 512) lstmmodel_changes.epochs = /*epochs*/ ctx[9];
    			if (dirty[0] & /*selectedOptimizer*/ 4096) lstmmodel_changes.selectedOptimizer = /*selectedOptimizer*/ ctx[12];
    			if (dirty[0] & /*learningRate*/ 8192) lstmmodel_changes.learningRate = /*learningRate*/ ctx[13];
    			if (dirty[0] & /*neuronCount*/ 16384) lstmmodel_changes.neuronCount = /*neuronCount*/ ctx[14];
    			lstmmodel.$set(lstmmodel_changes);
    			const configui_changes = {};
    			if (dirty[0] & /*modelIsWorking*/ 4) configui_changes.disabled = /*modelIsWorking*/ ctx[2];

    			if (!updating_name && dirty[0] & /*modelName*/ 2) {
    				updating_name = true;
    				configui_changes.name = /*modelName*/ ctx[1];
    				add_flush_callback(() => updating_name = false);
    			}

    			if (!updating_batchSize && dirty[0] & /*batchSize*/ 256) {
    				updating_batchSize = true;
    				configui_changes.batchSize = /*batchSize*/ ctx[8];
    				add_flush_callback(() => updating_batchSize = false);
    			}

    			if (!updating_epochs && dirty[0] & /*epochs*/ 512) {
    				updating_epochs = true;
    				configui_changes.epochs = /*epochs*/ ctx[9];
    				add_flush_callback(() => updating_epochs = false);
    			}

    			if (!updating_hiddenLayerCount && dirty[0] & /*hiddenLayerCount*/ 1024) {
    				updating_hiddenLayerCount = true;
    				configui_changes.hiddenLayerCount = /*hiddenLayerCount*/ ctx[10];
    				add_flush_callback(() => updating_hiddenLayerCount = false);
    			}

    			if (!updating_activationFunction && dirty[0] & /*activationFunction*/ 2048) {
    				updating_activationFunction = true;
    				configui_changes.activationFunction = /*activationFunction*/ ctx[11];
    				add_flush_callback(() => updating_activationFunction = false);
    			}

    			if (!updating_selectedOptimizer && dirty[0] & /*selectedOptimizer*/ 4096) {
    				updating_selectedOptimizer = true;
    				configui_changes.selectedOptimizer = /*selectedOptimizer*/ ctx[12];
    				add_flush_callback(() => updating_selectedOptimizer = false);
    			}

    			if (!updating_learningRate && dirty[0] & /*learningRate*/ 8192) {
    				updating_learningRate = true;
    				configui_changes.learningRate = /*learningRate*/ ctx[13];
    				add_flush_callback(() => updating_learningRate = false);
    			}

    			if (!updating_neuronCount && dirty[0] & /*neuronCount*/ 16384) {
    				updating_neuronCount = true;
    				configui_changes.neuronCount = /*neuronCount*/ ctx[14];
    				add_flush_callback(() => updating_neuronCount = false);
    			}

    			configui.$set(configui_changes);
    			const textfield0_changes = {};

    			if (!updating_value && dirty[0] & /*inputSize*/ 32768) {
    				updating_value = true;
    				textfield0_changes.value = /*inputSize*/ ctx[15];
    				add_flush_callback(() => updating_value = false);
    			}

    			textfield0.$set(textfield0_changes);
    			const textfield1_changes = {};

    			if (!updating_value_1 && dirty[0] & /*maxResultSize*/ 64) {
    				updating_value_1 = true;
    				textfield1_changes.value = /*maxResultSize*/ ctx[6];
    				add_flush_callback(() => updating_value_1 = false);
    			}

    			textfield1.$set(textfield1_changes);
    			const select_changes = {};
    			if (dirty[0] & /*trainingDataSets*/ 65536) select_changes.items = /*trainingDataSets*/ ctx[16];
    			if (dirty[0] & /*modelIsWorking*/ 4) select_changes.disabled = /*modelIsWorking*/ ctx[2];
    			select.$set(select_changes);
    			const textfield2_changes = {};

    			if (!updating_value_2 && dirty[0] & /*previewText*/ 262144) {
    				updating_value_2 = true;
    				textfield2_changes.value = /*previewText*/ ctx[18];
    				add_flush_callback(() => updating_value_2 = false);
    			}

    			textfield2.$set(textfield2_changes);
    			const button_changes = {};
    			if (dirty[0] & /*trainingData, modelIsWorking*/ 131076) button_changes.disabled = !/*trainingData*/ ctx[17] || /*modelIsWorking*/ ctx[2];

    			if (dirty[1] & /*$$scope*/ 2097152) {
    				button_changes.$$scope = { dirty, ctx };
    			}

    			button.$set(button_changes);
    			let previous_block_index = current_block_type_index;
    			current_block_type_index = select_block_type(ctx);

    			if (current_block_type_index === previous_block_index) {
    				if_blocks[current_block_type_index].p(ctx, dirty);
    			} else {
    				group_outros();

    				transition_out(if_blocks[previous_block_index], 1, 1, () => {
    					if_blocks[previous_block_index] = null;
    				});

    				check_outros();
    				if_block = if_blocks[current_block_type_index];

    				if (!if_block) {
    					if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
    					if_block.c();
    				} else {
    					if_block.p(ctx, dirty);
    				}

    				transition_in(if_block, 1);
    				if_block.m(div3, t16);
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(lstmmodel.$$.fragment, local);
    			transition_in(configui.$$.fragment, local);
    			transition_in(textfield0.$$.fragment, local);
    			transition_in(textfield1.$$.fragment, local);
    			transition_in(select.$$.fragment, local);
    			transition_in(textfield2.$$.fragment, local);
    			transition_in(button.$$.fragment, local);
    			transition_in(if_block);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(lstmmodel.$$.fragment, local);
    			transition_out(configui.$$.fragment, local);
    			transition_out(textfield0.$$.fragment, local);
    			transition_out(textfield1.$$.fragment, local);
    			transition_out(select.$$.fragment, local);
    			transition_out(textfield2.$$.fragment, local);
    			transition_out(button.$$.fragment, local);
    			transition_out(if_block);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(header);
    			if (detaching) detach_dev(t4);
    			if (detaching) detach_dev(main);
    			/*lstmmodel_binding*/ ctx[33](null);
    			destroy_component(lstmmodel);
    			destroy_component(configui);
    			destroy_component(textfield0);
    			destroy_component(textfield1);
    			destroy_component(select);
    			destroy_component(textfield2);
    			destroy_component(button);
    			if_blocks[current_block_type_index].d();
    			if (detaching) detach_dev(t17);
    			if (detaching) detach_dev(footer);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    async function loadTrainingData(url) {
    	const dataResponse = await fetch(url);
    	const data = await dataResponse.text();
    	return data;
    }

    function instance($$self, $$props, $$invalidate) {
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots("App", slots, []);
    	let taskTitle = "Language Model mit RNN";
    	let taskNumber = 4;
    	let labelSettings = "Settings";
    	let labelWordsForPrediction = "words used for predicting";
    	let labelShowResults = "Show results";
    	let labelDatasetsTitle = "Datasets";
    	let labelDatasetsText = "Preview Text";
    	let labelPredictionInputTitle = "Predicting Input";
    	let labelPredictionInput = "Input";
    	let labelPredictedItemsTitle = "Predicted Items";
    	let textWaitForModel = "creating/training model";

    	// Props
    	let model;

    	let modelName = "LSTM Model";
    	let modelIsWorking = false;
    	let predicting = false;
    	let input = "";
    	let inputPrediction = "";
    	let predictedItems = [];
    	let maxResultSize = 10;
    	let trained = false;

    	// initial Config
    	let batchSize = 512; // Neuronen min 32 max 512

    	let epochs = 10; // Trainings Epochen 50 iterations
    	let hiddenLayerCount = 5; // Anzahl der hidden Layer
    	let activationFunction = "relu";
    	let selectedOptimizer = "adam"; // Optimizer
    	let learningRate = 0.01; // Lernrate
    	let neuronCount = 50;
    	let inputSize = 3;

    	// Data
    	let trainingDataSets, trainingData, previewText;

    	// Documentation
    	let mdUrl = "./files/documentation.md";

    	// lifecycle functions
    	onMount(async () => {
    		let dataset1 = await loadTrainingData("./data/plenarprotokoll_230_20.05.2021.txt");
    		let dataset2 = await loadTrainingData("./data/test_data.txt");
    		let dataPreview = dataset1.slice(0, 300) + " ...";

    		$$invalidate(16, trainingDataSets = [
    			{
    				value: 0,
    				text: `100% (${parseInt(dataset1.length)} Characters) - Plenarprotokoll 20.05.2021`,
    				data: dataset1,
    				dataPreview
    			},
    			{
    				value: 1,
    				text: `50% (${parseInt(dataset1.length / 100 * 50)} Characters) - Plenarprotokoll 20.05.2021`,
    				data: dataset1.slice(0, parseInt(dataset1.length / 100 * 50)),
    				dataPreview
    			},
    			{
    				value: 2,
    				text: `25% (${parseInt(dataset1.length / 100 * 25)} Characters) - Plenarprotokoll 20.05.2021`,
    				data: dataset1.slice(0, parseInt(dataset1.length / 100 * 25)),
    				dataPreview
    			},
    			{
    				value: 3,
    				text: `10% (${parseInt(dataset1.length / 100 * 10)} Characters) - Plenarprotokoll 20.05.2021`,
    				data: dataset1.slice(0, parseInt(dataset1.length / 100 * 10)),
    				dataPreview
    			},
    			{
    				value: 4,
    				text: `1% (${parseInt(dataset1.length / 100 * 1)} Characters) - Plenarprotokoll 20.05.2021`,
    				data: dataset1.slice(0, parseInt(dataset1.length / 100 * 1)),
    				dataPreview
    			},
    			{
    				value: 5,
    				text: `Test Data (${parseInt(dataset1.length / 100 * 1)} Characters)`,
    				data: dataset2,
    				dataPreview: dataset2
    			}
    		]);
    	});

    	let train = async () => {
    		if (trainingData) {
    			await model.train(trainingData);
    			$$invalidate(7, trained = true);
    		}
    	};

    	let predict = async input => {
    		let aInput = input.split(" ");
    		let result = [];

    		if (aInput.length > 3) {
    			if (aInput[aInput.length - 1]) {
    				aInput[aInput.length - 1];
    			}

    			// Filter
    			// console.log(sInputFilter);
    			aInput = aInput.slice(0, aInput.length - 1);

    			result = await model.predict(aInput);
    		}

    		return result;
    	};

    	let predictInput = async e => {
    		if (input) {
    			$$invalidate(3, predicting = true);
    			let results = await predict(input);
    			$$invalidate(5, predictedItems = results.slice(0, maxResultSize).map(r => ({ text: r.word, subheading: r.acc })));
    			$$invalidate(3, predicting = false);
    		} else {
    			inputPrediction = "";
    		}
    	};

    	let selectItem = async e => {
    		let bEndsWithSpace = input.endsWith(" ");
    		let selectedItem = e.detail;

    		if (bEndsWithSpace) {
    			$$invalidate(4, input += selectedItem);
    		} else {
    			let aInput = input.split(" ");
    			aInput[aInput.length - 1] = selectedItem;
    			$$invalidate(4, input = aInput.join(" "));
    		}

    		$$invalidate(5, predictedItems = []);
    	};

    	const writable_props = [];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== "$$") console_1.warn(`<App> was created with unknown prop '${key}'`);
    	});

    	function lstmmodel_binding($$value) {
    		binding_callbacks[$$value ? "unshift" : "push"](() => {
    			model = $$value;
    			$$invalidate(0, model);
    		});
    	}

    	const predicting_handler = e => $$invalidate(2, modelIsWorking = e.detail);
    	const training_handler = e => $$invalidate(2, modelIsWorking = e.detail);

    	function configui_name_binding(value) {
    		modelName = value;
    		$$invalidate(1, modelName);
    	}

    	function configui_batchSize_binding(value) {
    		batchSize = value;
    		$$invalidate(8, batchSize);
    	}

    	function configui_epochs_binding(value) {
    		epochs = value;
    		$$invalidate(9, epochs);
    	}

    	function configui_hiddenLayerCount_binding(value) {
    		hiddenLayerCount = value;
    		$$invalidate(10, hiddenLayerCount);
    	}

    	function configui_activationFunction_binding(value) {
    		activationFunction = value;
    		$$invalidate(11, activationFunction);
    	}

    	function configui_selectedOptimizer_binding(value) {
    		selectedOptimizer = value;
    		$$invalidate(12, selectedOptimizer);
    	}

    	function configui_learningRate_binding(value) {
    		learningRate = value;
    		$$invalidate(13, learningRate);
    	}

    	function configui_neuronCount_binding(value) {
    		neuronCount = value;
    		$$invalidate(14, neuronCount);
    	}

    	function textfield0_value_binding(value) {
    		inputSize = value;
    		$$invalidate(15, inputSize);
    	}

    	function textfield1_value_binding(value) {
    		maxResultSize = value;
    		$$invalidate(6, maxResultSize);
    	}

    	const change_handler = v => {
    		$$invalidate(17, trainingData = trainingDataSets[v.detail].data);
    		$$invalidate(18, previewText = trainingDataSets[v.detail].dataPreview);
    		console.log(v.detail);
    	};

    	function textfield2_value_binding(value) {
    		previewText = value;
    		$$invalidate(18, previewText);
    	}

    	function textfield_value_binding(value) {
    		input = value;
    		$$invalidate(4, input);
    	}

    	$$self.$capture_state = () => ({
    		TextField,
    		Select,
    		Button,
    		ProgressCircular,
    		List,
    		onMount,
    		LSTMModel,
    		ConfigUI,
    		taskTitle,
    		taskNumber,
    		labelSettings,
    		labelWordsForPrediction,
    		labelShowResults,
    		labelDatasetsTitle,
    		labelDatasetsText,
    		labelPredictionInputTitle,
    		labelPredictionInput,
    		labelPredictedItemsTitle,
    		textWaitForModel,
    		model,
    		modelName,
    		modelIsWorking,
    		predicting,
    		input,
    		inputPrediction,
    		predictedItems,
    		maxResultSize,
    		trained,
    		batchSize,
    		epochs,
    		hiddenLayerCount,
    		activationFunction,
    		selectedOptimizer,
    		learningRate,
    		neuronCount,
    		inputSize,
    		trainingDataSets,
    		trainingData,
    		previewText,
    		mdUrl,
    		loadTrainingData,
    		train,
    		predict,
    		predictInput,
    		selectItem
    	});

    	$$self.$inject_state = $$props => {
    		if ("taskTitle" in $$props) $$invalidate(19, taskTitle = $$props.taskTitle);
    		if ("taskNumber" in $$props) $$invalidate(20, taskNumber = $$props.taskNumber);
    		if ("labelSettings" in $$props) $$invalidate(21, labelSettings = $$props.labelSettings);
    		if ("labelWordsForPrediction" in $$props) $$invalidate(22, labelWordsForPrediction = $$props.labelWordsForPrediction);
    		if ("labelShowResults" in $$props) $$invalidate(23, labelShowResults = $$props.labelShowResults);
    		if ("labelDatasetsTitle" in $$props) $$invalidate(24, labelDatasetsTitle = $$props.labelDatasetsTitle);
    		if ("labelDatasetsText" in $$props) $$invalidate(25, labelDatasetsText = $$props.labelDatasetsText);
    		if ("labelPredictionInputTitle" in $$props) $$invalidate(26, labelPredictionInputTitle = $$props.labelPredictionInputTitle);
    		if ("labelPredictionInput" in $$props) $$invalidate(27, labelPredictionInput = $$props.labelPredictionInput);
    		if ("labelPredictedItemsTitle" in $$props) $$invalidate(28, labelPredictedItemsTitle = $$props.labelPredictedItemsTitle);
    		if ("textWaitForModel" in $$props) textWaitForModel = $$props.textWaitForModel;
    		if ("model" in $$props) $$invalidate(0, model = $$props.model);
    		if ("modelName" in $$props) $$invalidate(1, modelName = $$props.modelName);
    		if ("modelIsWorking" in $$props) $$invalidate(2, modelIsWorking = $$props.modelIsWorking);
    		if ("predicting" in $$props) $$invalidate(3, predicting = $$props.predicting);
    		if ("input" in $$props) $$invalidate(4, input = $$props.input);
    		if ("inputPrediction" in $$props) inputPrediction = $$props.inputPrediction;
    		if ("predictedItems" in $$props) $$invalidate(5, predictedItems = $$props.predictedItems);
    		if ("maxResultSize" in $$props) $$invalidate(6, maxResultSize = $$props.maxResultSize);
    		if ("trained" in $$props) $$invalidate(7, trained = $$props.trained);
    		if ("batchSize" in $$props) $$invalidate(8, batchSize = $$props.batchSize);
    		if ("epochs" in $$props) $$invalidate(9, epochs = $$props.epochs);
    		if ("hiddenLayerCount" in $$props) $$invalidate(10, hiddenLayerCount = $$props.hiddenLayerCount);
    		if ("activationFunction" in $$props) $$invalidate(11, activationFunction = $$props.activationFunction);
    		if ("selectedOptimizer" in $$props) $$invalidate(12, selectedOptimizer = $$props.selectedOptimizer);
    		if ("learningRate" in $$props) $$invalidate(13, learningRate = $$props.learningRate);
    		if ("neuronCount" in $$props) $$invalidate(14, neuronCount = $$props.neuronCount);
    		if ("inputSize" in $$props) $$invalidate(15, inputSize = $$props.inputSize);
    		if ("trainingDataSets" in $$props) $$invalidate(16, trainingDataSets = $$props.trainingDataSets);
    		if ("trainingData" in $$props) $$invalidate(17, trainingData = $$props.trainingData);
    		if ("previewText" in $$props) $$invalidate(18, previewText = $$props.previewText);
    		if ("mdUrl" in $$props) $$invalidate(29, mdUrl = $$props.mdUrl);
    		if ("train" in $$props) $$invalidate(30, train = $$props.train);
    		if ("predict" in $$props) predict = $$props.predict;
    		if ("predictInput" in $$props) $$invalidate(31, predictInput = $$props.predictInput);
    		if ("selectItem" in $$props) $$invalidate(32, selectItem = $$props.selectItem);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	return [
    		model,
    		modelName,
    		modelIsWorking,
    		predicting,
    		input,
    		predictedItems,
    		maxResultSize,
    		trained,
    		batchSize,
    		epochs,
    		hiddenLayerCount,
    		activationFunction,
    		selectedOptimizer,
    		learningRate,
    		neuronCount,
    		inputSize,
    		trainingDataSets,
    		trainingData,
    		previewText,
    		taskTitle,
    		taskNumber,
    		labelSettings,
    		labelWordsForPrediction,
    		labelShowResults,
    		labelDatasetsTitle,
    		labelDatasetsText,
    		labelPredictionInputTitle,
    		labelPredictionInput,
    		labelPredictedItemsTitle,
    		mdUrl,
    		train,
    		predictInput,
    		selectItem,
    		lstmmodel_binding,
    		predicting_handler,
    		training_handler,
    		configui_name_binding,
    		configui_batchSize_binding,
    		configui_epochs_binding,
    		configui_hiddenLayerCount_binding,
    		configui_activationFunction_binding,
    		configui_selectedOptimizer_binding,
    		configui_learningRate_binding,
    		configui_neuronCount_binding,
    		textfield0_value_binding,
    		textfield1_value_binding,
    		change_handler,
    		textfield2_value_binding,
    		textfield_value_binding
    	];
    }

    class App extends SvelteComponentDev {
    	constructor(options) {
    		super(options);
    		init(this, options, instance, create_fragment, safe_not_equal, {}, [-1, -1]);

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "App",
    			options,
    			id: create_fragment.name
    		});
    	}
    }

    const app = new App({
    	target: document.body,
    	props: {}
    });

    return app;

}());
//# sourceMappingURL=bundle.js.map
