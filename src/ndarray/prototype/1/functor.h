	// make functor

	template<typename F, F f, typename... AA> struct functor_b{
	    __HOSTDEVICE__ void operator()(AA&&... aa) const {
	        f(aa...);
	    }
	};

	template<typename F, F f> struct functor_parse{
	    template<typename... AA>
	    static functor_b<F, f, AA...> make_functor(void (fp)(AA...) ){
	        return functor_b<F, f, AA...>();
	    }
	};

	template<typename F, typename std::decay<F>::type f> struct functor : public decltype ( functor_parse< typename std::decay<F>::type ,f>::make_functor(f) ) {
	};

