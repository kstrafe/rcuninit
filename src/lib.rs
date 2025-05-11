//! Defines `RcUninit`, an `Rc` with deferred initialization.
//!
//! [RcUninit] solves two problems with [Rc::new_cyclic], namely:
//!
//! 1. Inability to use across await points - `new_cyclic` takes a closure that
//!    is not async.
//! 2. Instantiation of complex cyclic structures is cumbersome and has a high
//!    mental overhead.
//!
//! # Examples #
//!
//! ```
//! use rcuninit::RcUninit;
//! use std::rc::Rc;
//!
//! // Must be called at least once per program invocation.
//! unsafe {
//!     rcuninit::check_sanity();
//! }
//!
//! let rcuninit = RcUninit::new();
//!
//! // Acquire a weak pointer.
//! let weak = rcuninit.weak();
//!
//! assert!(weak.upgrade().is_none());
//!
//! // Now initialize, returns an Rc and makes associated Weaks upgradable.
//! let strong = rcuninit.init(String::from("lorem ipsum"));
//!
//! assert_eq!(*weak.upgrade().unwrap(), "lorem ipsum");
//! assert!(Rc::ptr_eq(&strong, &weak.upgrade().unwrap()));
//! ```
//!
//! Here's an example of initialization across an `await` point. This is not
//! possible with `Rc::new_cyclic`.
//!
//! ```
//! use rcuninit::RcUninit;
//! use std::{future::poll_fn, rc::Rc, task::Poll};
//!
//! async fn f() {
//!     let rcuninit = RcUninit::new();
//!     let weak = rcuninit.weak();
//!
//!     poll_fn(|_| Poll::Ready(())).await;
//!
//!     let strong = rcuninit.init(String::from("lorem ipsum"));
//!     assert_eq!(*weak.upgrade().unwrap(), "lorem ipsum");
//! }
//! ```
//!
//! # Complex Cyclic Structures #
//!
//! The other issue mentioned regarding `Rc::new_cyclic` is its cumbersome
//! nature when attempting to declare complex cyclic structures. Suppose we want
//! to construct a structure `A => B => C -> A`, where `=>` and `->` denote a
//! strong and weak pointer, respectively. The following two examples show the
//! difference between [RcUninit] and [Rc::new_cyclic].
//!
//! ```
//! use rcuninit::RcUninit;
//! use std::rc::{Rc, Weak};
//!
//! unsafe {
//!     rcuninit::check_sanity();
//! }
//!
//! let a_un = RcUninit::new();
//! let b_un = RcUninit::new();
//! let c_un = RcUninit::new();
//!
//! let c = c_un.init(C { a: a_un.weak() });
//! let b = b_un.init(B { c });
//! let a = a_un.init(A { b });
//!
//! assert!(Rc::ptr_eq(&a.b.c.a.upgrade().unwrap(), &a));
//!
//! struct A {
//!     b: Rc<B>,
//! }
//! struct B {
//!     c: Rc<C>,
//! }
//! struct C {
//!     a: Weak<A>,
//! }
//! ```
//!
//! Now compare this to the construction of the same structure using
//! `Rc::new_cyclic`. Structure definitions hidden for brevity, but they are
//! identical as above.
//!
//! ```
//! use std::rc::{Rc, Weak};
//!
//! let mut b: Option<Rc<B>> = None;
//! let mut c: Option<Rc<C>> = None;
//!
//! let a = Rc::new_cyclic(|a_weak| {
//!     let b_rc = Rc::new_cyclic(|b_weak| {
//!         let c_rc = Rc::new(C { a: a_weak.clone() });
//!         c = Some(c_rc.clone());
//!         B { c: c_rc }
//!     });
//!
//!     b = Some(b_rc.clone());
//!     A { b: b_rc }
//! });
//!
//! let b = b.unwrap();
//! let c = c.unwrap();
//!
//! assert!(Rc::ptr_eq(&a.b.c.a.upgrade().unwrap(), &a));
//!
//! # struct A { b: Rc<B> }
//! # struct B { c: Rc<C> }
//! # struct C { a: Weak<A> }
//! ```
//!
//! Note that we store `a`, `b`, and `c` into variables because we imagine the
//! code having some use for them later on.
//!
//! One alternative in the last example
//! is to implement `get` methods on the structs to get the values out, but that
//! is still noisy and cumbersome. This example only grows more difficult to
//! mentally process when there are more pointers, eventually becoming unwieldy
//! to deal with.
//!
//! Another option here is to use interior mutability and use `set_weak_pointer`
//! on these structs to get the desired effect, but that requires us to set
//! pointers after initialization which is prone to the inadvertent creation of
//! reference cycles.
//!
//! # Sanity Checking #
//!
//! This crate makes assumptions about how data inside [Rc] is laid out. As of
//! writing this documentation, `Rc` holds a pointer to `RcInner`.
//!
//! ```no_run
//! # use std::cell::Cell;
//! #[repr(C)]
//! struct RcInner<T: ?Sized> {
//!     strong: Cell<usize>,
//!     weak: Cell<usize>,
//!     value: T,
//! }
//! ```
//!
//! Internally, we consume an `Rc<MaybeUninit<T>>` via [Rc::into_raw], which
//! gives us a pointer to `value` field. We then calculate the offsets manually
//! (accounting for padding) to reach `strong` and `weak` such that these fields
//! can be manipulated directly to serve the purposes of [RcUninit].
//!
//! To guard against future changes in the standard library, we must perform a
//! sanity check every time the program is run. This is to test whether the
//! values we are reaching into are actually located where we believe they
//! should be located.
//!
//! This sanity checking is performed by calling:
//!
//! ```
//! use rcuninit::check_sanity;
//!
//! unsafe {
//!     check_sanity();
//! }
//! ```
//!
//! See [check_sanity] for more details.
//!
//! # Native Rust Support #
//!
//! There are ongoing discussions on getting `RcUninit` and related features
//! (`UniqueRc`) into std. This crate will be superceded once `RcUninit` is
//! natively supported.
//!
//! - <https://github.com/rust-lang/libs-team/issues/90>
//! - <https://github.com/rust-lang/rust/issues/112566>
#![feature(alloc_layout_extra)]
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{MaybeUninit, forget, transmute},
    rc::{Rc, Weak},
    sync::atomic::{AtomicBool, Ordering},
};

static SANITY_CHECKED: AtomicBool = AtomicBool::new(false);

/// Check whether the assumptions about the internals of [Rc] made in this
/// library are correct.
///
/// Note that we can't verify exactly whether we are out-of-sync with the
/// current version of std, instead, we perform some sanity testing that should
/// give us a "good enough" indicator that our assumptions are correct.
///
/// As such, this function is marked unsafe, because it can invoke undefined
/// behavior if our assumptions are wrong.
///
/// Once this check has passed, a global variable is set that allows the
/// creation of [RcUninit]. Note that even with sanity tests, this might still
/// invoke undefined behavior if some detail is not caught. [Rc] could decide to
/// reorder fields at will after some call, or do something else that
/// invalidates the assumptions made in this library. As such, use of
/// `RcUninit` _could_ be unsound, however unlikely. The functions on `RcUninit`
/// are thus not marked as unsafe as `check_sanity` intends to provide
/// sufficient shielding against potential undefined behavior.
///
/// # Safety #
///
/// This function is only sound if the standard library used with this crate
/// matches this crate's assumptions about [Rc], as such, it's up to the caller
/// to inspect the standard library to confirm that it matches what this crate
/// expects.
///
/// Here are the conditions that need to be satisfied by [Rc] and [Weak].
///
/// 1. `RcInner` definitions inside `std::rc` and this crate must match exactly.
/// 2. The pointer to `RcInner` that [Rc] constructs must be `*mut`.
/// 3. The value pointer that [Rc::into_raw] returns must point to the value
///    inside `RcInner`.
/// 4. Offset calculations used by [Rc::increment_strong_count] and
///    [Rc::decrement_strong_count] must match with how this crate calculates
///    offsets from the value in `RcInner` to the counts.
/// 5. All operations that [Weak] exposes must not invalidate the pointer to
///    `RcInner`.
pub unsafe fn check_sanity() {
    {
        let uninit = RcUninit::new_assume_sane();

        let weak = uninit.weak();
        assert_eq!(0, Weak::strong_count(&weak));
        // Returns 0 when the strong count is 0, but internally a true weak count is
        // stored.
        assert_eq!(0, Weak::weak_count(&weak));

        let inner_ptr = uninit.ptr();
        let inner = unsafe { get_inner(inner_ptr) };

        assert_eq!(inner.strong.get(), 0);
        assert_eq!(inner.weak.get(), 3);

        let weak2 = weak.clone();

        assert_eq!(inner.strong.get(), 0);
        assert_eq!(inner.weak.get(), 4);

        assert_eq!(0, Weak::strong_count(&weak));
        // Returns 0 when the strong count is 0, but internally a true weak count is
        // stored.
        assert_eq!(0, Weak::weak_count(&weak));

        assert!(weak.upgrade().is_none());
        assert!(weak2.upgrade().is_none());

        // init acquires a mutable references to inner, so we must drop our inner and
        // then reacquire it to avoid mutable aliasing.
        let _ = inner;
        let rc = uninit.init(123);
        let inner = unsafe { get_inner(inner_ptr) };

        assert_eq!(inner.strong.get(), 1);
        // RcUninit's internal Weak is dropped, so the count has decreased.
        assert_eq!(inner.weak.get(), 3);

        let rc2 = rc.clone();
        assert_eq!(inner.strong.get(), 2);
        assert_eq!(inner.weak.get(), 3);
        assert_eq!(2, Weak::strong_count(&weak));
        assert_eq!(2, Weak::weak_count(&weak)); // NOTE: Weak count is 1 less than the actual value
        // stored.

        assert!(Rc::ptr_eq(&rc, &rc2));
        assert!(Rc::ptr_eq(&rc, &weak.upgrade().unwrap()));
        assert!(Rc::ptr_eq(&rc, &weak2.upgrade().unwrap()));

        drop(rc2);
        assert_eq!(inner.strong.get(), 1);
        assert_eq!(inner.weak.get(), 3);
        assert_eq!(1, Weak::strong_count(&weak));
        assert_eq!(2, Weak::weak_count(&weak));

        drop(weak2);
        assert_eq!(inner.strong.get(), 1);
        assert_eq!(inner.weak.get(), 2);
        assert_eq!(1, Weak::strong_count(&weak));
        assert_eq!(1, Weak::weak_count(&weak));

        drop(rc);
        assert_eq!(inner.strong.get(), 0);
        assert_eq!(inner.weak.get(), 1);
        assert_eq!(0, Weak::strong_count(&weak));
        assert_eq!(0, Weak::weak_count(&weak));
    }

    SANITY_CHECKED.store(true, Ordering::Relaxed);
}

/// RcInner type used in std, this should reflect the exact same type and
/// ordering.
#[repr(C)]
struct RcInner<T: ?Sized> {
    strong: Cell<usize>,
    weak: Cell<usize>,
    value: T,
}

/// Defer initialization of [Rc] while providing [Weak] pointers to the
/// allocation.
pub struct RcUninit<T> {
    ptr: *const MaybeUninit<T>,
    weak: Weak<T>,
}

impl<T> RcUninit<T> {
    /// Creates a new `RcUninit`.
    ///
    /// # Panics #
    ///
    /// Panics if [check_sanity] has not been called at least once.
    ///
    /// # Examples #
    ///
    /// ```
    /// use rcuninit::RcUninit;
    /// unsafe {
    ///     rcuninit::check_sanity();
    /// }
    /// RcUninit::<i32>::new();
    /// ```
    pub fn new() -> Self {
        assert!(
            SANITY_CHECKED.load(Ordering::Relaxed),
            "must call check_sanity() before using this library"
        );

        Self::new_assume_sane()
    }

    fn ptr(&self) -> *const MaybeUninit<T> {
        self.ptr
    }

    fn new_assume_sane() -> Self {
        let rc: Rc<MaybeUninit<T>> = Rc::new(MaybeUninit::uninit());

        // SAFETY: Transmuting from a Weak<MaybeUninit<T>> into a Weak<T> should safe
        // since MaybeUninit is #[repr(transparent)], and the user will not be
        // able to upgrade this weak pointer until init is called to access
        // uninitialized data since we set the strong count to zero.
        let weak: Weak<T> = unsafe { transmute(Rc::downgrade(&rc)) };

        let ptr = Rc::into_raw(rc);

        // SAFETY: Subtracting the offset should point us to the actual RcInner used in
        // std, which should have the same layout as our RcInner.
        let inner: &RcInner<MaybeUninit<T>> = unsafe { get_inner(ptr) };

        // The following checks can't guarantee soundness since they occur after
        // acquiring RcInner, but they will stop the program from running if we
        // point to nonsensical values. Note that if undefined behavior was
        // triggered, then these values can be as expected.
        //
        // We make sure that we can read `1` from this location.
        assert_eq!(inner.strong.get(), 1);

        // We make sure that we can read `2` from this location, since we have created a
        // Weak, and all Rcs share a single shared weak reference by themselves.
        assert_eq!(inner.weak.get(), 2);

        // Set the strong count to 0, preventing upgrades of Weak to this allocation.
        // We do not use Rc::decrement_strong_count since that causes weaks to be
        // upgradable because it reconstructs the "strong weak" pointer which
        // drops the contained value.
        inner.strong.set(0);

        Self { ptr, weak }
    }

    /// Returns a [Weak] pointer that cannot be upgraded until [RcUninit::init]
    /// is called.
    ///
    /// # Examples #
    ///
    /// ```
    /// use rcuninit::RcUninit;
    /// use std::rc::Weak;
    ///
    /// unsafe {
    ///     rcuninit::check_sanity();
    /// }
    ///
    /// let uninit = RcUninit::<i32>::new();
    /// let weak: Weak<i32> = uninit.weak();
    /// ```
    pub fn weak(&self) -> Weak<T> {
        // The user won't be able to upgrade this weak pointer until `init` gets called,
        // since the strong count was set to zero in RcUninit::new.
        self.weak.clone()
    }

    /// Initializes the pointer.
    ///
    /// # Examples #
    ///
    /// ```
    /// use rcuninit::RcUninit;
    /// use std::rc::Rc;
    ///
    /// unsafe {
    ///     rcuninit::check_sanity();
    /// }
    ///
    /// let uninit = RcUninit::new();
    /// let rc: Rc<i32> = uninit.init(123);
    /// ```
    pub fn init(self, value: T) -> Rc<T> {
        let ptr = self.ptr;

        // SAFETY: There are no current borrows of the RcInner, so we can dereference as
        // mutable. This should point to a valid RcInner since the layouts are
        // the same.
        let inner: &mut RcInner<MaybeUninit<T>> = unsafe { get_inner_mut(ptr) };

        // Perform some extra sanity checks, at this point, we should point to a valid
        // `RcInner`, the strong count must be zero, and the weak count can be
        // anything above or equal to 2, since we have the initial Rc, as well
        // as the Weak stored by this struct.
        assert_eq!(inner.strong.get(), 0);
        assert!(inner.weak.get() >= 2);

        inner.strong.set(1);

        // We decrement the weak reference count here because we are about to forget
        // self which skips running drop for self, which would run drop for
        // weak.
        inner.weak.set(inner.weak.get() - 1);

        inner.value.write(value);

        // SAFETY: Since the pointer was created from Rc::into_raw, this call should be
        // sound.
        let rc = unsafe { Rc::from_raw(ptr) };

        // Skip running the destructor since re-acquire the Rc and return it. Rc's
        // destructor will perform the cleanup instead.
        forget(self);

        // SAFETY: We convert Rc<MaybeUninit<T>> into Rc<T>, which should be sound given
        // that MaybeUninit has now been written to.
        unsafe { rc.assume_init() }
    }

    /// Initializes the pointer and provides the weak pointer.
    ///
    /// # Examples #
    ///
    /// ```
    /// use rcuninit::RcUninit;
    /// use std::rc::Rc;
    ///
    /// unsafe {
    ///     rcuninit::check_sanity();
    /// }
    ///
    /// let uninit = RcUninit::new();
    /// let rc: Rc<i32> = uninit.init_with(|weak| {
    ///     // Do something with weak
    ///     123
    /// });
    /// ```
    pub fn init_with<F>(self, constructor: F) -> Rc<T>
    where
        F: FnOnce(Weak<T>) -> T,
    {
        let weak = self.weak();
        self.init(constructor(weak))
    }
}

impl<T> Default for RcUninit<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for RcUninit<T> {
    fn drop(&mut self) {
        let offset = data_offset::<T>();

        // SAFETY: Pointer should point to a valid RcInner<MaybeUninit<T>>. It's
        // important to declare this as a pointer to MaybeUninit<_> so we do not
        // run the desctructor since the data has not been initialized.
        let inner = unsafe { &mut *(self.ptr.byte_sub(offset) as *mut RcInner<MaybeUninit<T>>) };

        inner.strong.set(1);

        // SAFETY: Pointer was created with into_raw.
        unsafe { Rc::from_raw(self.ptr) };
    }
}

fn data_offset<T>() -> usize {
    // This should be correct, currently, std uses an intrinsic that we cannot use.
    data_offset_align(align_of::<T>())
}

#[inline]
fn data_offset_align(align: usize) -> usize {
    let layout = Layout::new::<RcInner<()>>();
    layout.size() + layout.padding_needed_for(align)
}

unsafe fn get_inner<'a, T>(ptr: *const MaybeUninit<T>) -> &'a RcInner<MaybeUninit<T>> {
    let offset = data_offset::<T>();

    // SAFETY: Subtracting the offset should point us to the actual RcInner used in
    // std, which should have the same layout as our RcInner.
    unsafe { &*(ptr.byte_sub(offset) as *const RcInner<MaybeUninit<T>>) }
}

unsafe fn get_inner_mut<'a, T>(ptr: *const MaybeUninit<T>) -> &'a mut RcInner<MaybeUninit<T>> {
    let offset = data_offset::<T>();

    // SAFETY: Subtracting the offset should point us to the actual RcInner used in
    // std, which should have the same layout as our RcInner.
    unsafe { &mut *(ptr.byte_sub(offset) as *mut RcInner<MaybeUninit<T>>) }
}

#[test]
fn basic() {
    unsafe { check_sanity() };

    let x = RcUninit::new();

    let weak = x.weak();
    assert!(weak.upgrade().is_none());

    let rc = x.init(123);
    assert_eq!(weak.upgrade().map(|x| *x), Some(123));

    drop(rc);

    assert!(weak.upgrade().is_none());
}

#[test]
fn dst() {
    unsafe { check_sanity() };

    let x = RcUninit::<i32>::new();

    let weak = x.weak();
    let weak: Weak<dyn std::fmt::Debug> = weak;
    assert!(weak.upgrade().is_none());

    let _rc = x.init(123);

    let upgraded = weak.upgrade();
    assert!(upgraded.is_some());

    assert_eq!("123", format!("{:?}", upgraded.unwrap()));
}

#[test]
fn zst() {
    unsafe { check_sanity() };

    let x = RcUninit::<()>::new();

    let weak = x.weak();
    assert!(weak.upgrade().is_none());

    let _rc = x.init(());

    let upgraded = weak.upgrade();
    assert!(upgraded.is_some());
}

#[test]
fn odd_sized_type() {
    unsafe { check_sanity() };

    let x = RcUninit::<[u8; 33]>::new();

    let weak = x.weak();
    assert!(weak.upgrade().is_none());

    let rc = x.init([137u8; 33]);

    let upgraded = weak.upgrade();
    assert!(upgraded.is_some());

    assert_eq!(33, rc.len());
    for value in *rc {
        assert_eq!(137u8, value);
    }
}

#[test]
fn panic_on_drop_not_initialized() {
    unsafe { check_sanity() };

    struct PanicOnDrop;
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!();
        }
    }

    RcUninit::<PanicOnDrop>::new();
}

#[test]
fn count_drops() {
    unsafe { check_sanity() };

    let counter = Rc::new(Cell::new(0));

    struct DropCounter(Rc<Cell<usize>>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    let uninit = RcUninit::<DropCounter>::new();
    assert_eq!(counter.get(), 0);

    uninit.weak();
    assert_eq!(counter.get(), 0);

    uninit.init(DropCounter(counter.clone()));
    assert_eq!(counter.get(), 1);

    let _ = RcUninit::<DropCounter>::new();
    assert_eq!(counter.get(), 1);
}
