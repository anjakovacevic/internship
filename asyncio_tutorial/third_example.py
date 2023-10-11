import api
import asyncio


async def main():
    task = asyncio.create_task(api.fetch_data())
    '''
    # it is better to crceate a task instead of a regular function or a statement 
    # because the task has more options, like:
    task.cancel()
    await asyncio.sleep(0.5)
    
    if task.cancelled():
        print(task.cancelled())
    '''
    # await asyncio.sleep(3)

    try:
        # If we want it to give up after 2 seconds
        # If the task is taking too long, there is a problem with communication, internet,...
        await asyncio.wait_for(task, timeout=0.001)

        '''Check if the task is done
        if task.done():
            print(task.result())  # Prints data
            '''
        '''
        result = await task
        print(result)
        '''
    except asyncio.CancelledError:
        print('CANCELLED: Request was cancelled...')
    except asyncio.TimeoutError:
        print('TIMEOUT: Request took too long...')


if __name__ == '__main__':
    asyncio.run(main())