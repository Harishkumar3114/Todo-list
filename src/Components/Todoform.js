    import React, { useState } from 'react'

    const Todoform = ({addprops}) => {
        const [value,Setvalue] = useState("")
        const handleChange = (e) => {
            e.preventDefault()
            addprops(value)
            Setvalue("")

        }
    return (
        <form onSubmit={handleChange} className='TodoForm'>
            <input type='text' className='todo-input' placeholder='What is the task today?' value={value} onChange={(e) => {
                Setvalue(e.target.value)
            }}/>
            <button type='submit' className='todo-btn'>Add task</button>
        </form>
    )
    }

    export default Todoform
