var dagcomponentfuncs = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

dagcomponentfuncs.Button = function (props) {
    const {setData, data} = props;

    function onClick() {
        setData(data); // Ensure the data is passed correctly
    }

    return React.createElement(
        'button',
        {
            onClick: onClick,
            className: props.className,
        },
        props.value
    );
};